"""
Generation engine for FLUX CUDA backend.

Handles image generation, parameter validation, and result processing.
"""

import warnings
import gc
import logging
from typing import Dict, Any, Tuple

# Import with fallback for standalone loading
try:
    from .flux_memory_manager import FluxMemoryManager, OperationType
    from .flux_pipeline_loader import FluxPipelineLoader
except ImportError:
    import importlib.util
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load FluxMemoryManager
    spec = importlib.util.spec_from_file_location("flux_memory_manager", os.path.join(current_dir, "flux_memory_manager.py"))
    memory_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory_module)
    FluxMemoryManager = memory_module.FluxMemoryManager
    OperationType = memory_module.OperationType
    
    # Load FluxPipelineLoader
    spec = importlib.util.spec_from_file_location("flux_pipeline_loader", os.path.join(current_dir, "flux_pipeline_loader.py"))
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    FluxPipelineLoader = pipeline_module.FluxPipelineLoader

logger = logging.getLogger(__name__)


class FluxGenerationEngine:
    """Manages FLUX image generation with proper parameter validation and error handling."""
    
    def __init__(self, shared_backend: Dict[str, Any], memory_manager: "FluxMemoryManager", pipeline_loader: "FluxPipelineLoader", config: "FluxConfig" = None):
        """Initialize with shared backend, memory manager, pipeline loader, and configuration."""
        self._shared_backend = shared_backend
        self._memory_manager = memory_manager
        self._pipeline_loader = pipeline_loader
        self._config = config
        self._first_run = True  # Track if this is the first model load
    
    def _should_do_aggressive_cleanup(self) -> bool:
        """Check if aggressive memory cleanup is enabled in config"""
        if self._config is None:
            return True  # Default enabled
        cleanup_config = self._config.get_memory_cleanup_config()
        return cleanup_config.get("enabled", True) and cleanup_config.get("pre_inference_cleanup", True)
    
    def _should_do_emergency_vae_fallback(self) -> bool:
        """Check if emergency VAE CPU fallback is enabled in config"""
        if self._config is None:
            return True  # Default enabled
        cleanup_config = self._config.get_memory_cleanup_config()
        return cleanup_config.get("enabled", True) and cleanup_config.get("emergency_vae_cpu_fallback", True)
    
    def generate_image(self, **kwargs):
        """Generate image directly (no generator)"""
        print(f"[FLUX DEBUG] ===== DiffusersFluxBackend.generate_image CALLED =====")
        print(f"[FLUX DEBUG] All kwargs keys: {list(kwargs.keys())}")
        print(f"[FLUX DEBUG] All kwargs: {kwargs}")
        
        # CRITICAL: Check prompt parameter immediately
        prompt_value = kwargs.get('prompt', 'MISSING_KEY')
        print(f"[FLUX DEBUG] PROMPT CHECK: '{prompt_value}' (type: {type(prompt_value)})")
        
        if not prompt_value or prompt_value == 'MISSING_KEY':
            print(f"[FLUX DEBUG] ❌ CRITICAL: Prompt is missing or empty!")
            print(f"[FLUX DEBUG] Available kwargs: {list(kwargs.keys())}")
            raise ValueError(f"Prompt parameter missing or empty. Got: '{prompt_value}'. Available kwargs: {list(kwargs.keys())}")
        
        # Use pre-loaded torch from shared backend
        torch = self._shared_backend['torch']
        
        model_id = kwargs['model_id']
        prompt = kwargs['prompt']
        width = kwargs['width']
        height = kwargs['height']
        steps = kwargs['steps']
        guidance = kwargs['guidance']
        seed = kwargs['seed']
        max_sequence_length = kwargs.get('max_sequence_length', 512)
        quantization = kwargs.get('quantization', 'none')
        system_constraints = kwargs.get('system_constraints', {})
        
        print(f"[FLUX DEBUG] Extracted quantization: '{quantization}' (type: {type(quantization)})")
        print(f"[FLUX DEBUG] Raw quantization from kwargs: {kwargs.get('quantization', 'NOT_FOUND')}")
        print(f"[FLUX DEBUG] Model ID: {model_id}")
        print(f"[FLUX DEBUG] System constraints keys: {list(system_constraints.keys()) if system_constraints else 'None'}")
        
        # Reserve working memory for inference operation
        working_memory_reserved = False
        if kwargs.get('width', 512) * kwargs.get('height', 512) > 1024 * 1024:
            # Large images need more working memory
            working_memory_reserved = self._memory_manager.reserve_working_memory(
                OperationType.LONG_CONTEXT, duration_seconds=600.0
            )
        else:
            working_memory_reserved = self._memory_manager.reserve_working_memory(
                OperationType.INFERENCE, duration_seconds=300.0
            )
        
        if not working_memory_reserved:
            print(f"[FLUX MEMORY] ⚠️ Could not reserve working memory - proceeding with caution")
        
        # Smart GPU memory check - only clear cache if needed
        should_clear_cache = self._should_clear_gpu_cache()
        if should_clear_cache:
            print(f"[FLUX MEMORY] Clearing GPU cache before model load...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.cuda.ipc_collect() if hasattr(torch.cuda, 'ipc_collect') else None
            torch.cuda.synchronize()
        else:
            print(f"[FLUX MEMORY] GPU cache is clean - skipping cache clear for faster startup")

        # Wait for allocator fragmentation to clear
        def _wait_for_gpu_free(min_free_gb: float = 4.0, timeout: float = 30):
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            import time
            start = time.time()
            while time.time() - start < timeout:
                free_gb = (total - (torch.cuda.memory_allocated(0)+ torch.cuda.memory_reserved(0)) / 1024**3)
                if free_gb >= min_free_gb:
                    print(f'[FLUX MEMORY] {free_gb:.1f}GB free after cleanup')
                    return
                time.sleep(1)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() if hasattr(torch.cuda,'ipc_collect') else None
            print('[FLUX MEMORY] Timeout waiting for free VRAM; proceeding anyway')
        if should_clear_cache:
            _wait_for_gpu_free()
        gc.collect()
        
        # Mark that we've done our first run
        self._first_run = False

        # Get pipeline with async loading to prevent Griptape timeouts
        try:
            pipeline_result = self._pipeline_loader.get_pipeline_with_progress(model_id, quantization, system_constraints)
            
            # Handle different result types
            if pipeline_result == "ASYNC_MEMORY_CHECK_NEEDED":
                raise RuntimeError("ASYNC_MEMORY_CHECK_NEEDED")
            elif str(type(pipeline_result).__name__) == 'generator':
                # It's a generator - need to iterate through it
                print("[FLUX LOADING] 🔄 Loading pipeline asynchronously...")
                pipeline = None
                for step in pipeline_result:
                    if isinstance(step, str):
                        print(f"[FLUX LOADING] {step}")
                        # Yield control back to Griptape during long operations
                        yield f"Pipeline loading: {step}"
                    else:
                        # Final pipeline result
                        pipeline = step
                        break
                        
                if not pipeline:
                    raise RuntimeError("Pipeline loading failed - no result returned")
            else:
                # Direct pipeline result (cached)
                pipeline = pipeline_result
                
        except RuntimeError as e:
            if "no kernel image" in str(e).lower():
                print("[FLUX FALLBACK] bitsandbytes kernel not available for this GPU – retrying full precision …")
                quantization = "none"
                fallback_result = self._pipeline_loader.get_pipeline_with_progress(model_id, "none", system_constraints)
                
                # Handle fallback result (could also be generator)
                if hasattr(fallback_result, '__iter__') and not isinstance(fallback_result, str):
                    print("[FLUX LOADING] 🔄 Loading fallback pipeline asynchronously...")
                    pipeline = None
                    for step in fallback_result:
                        if isinstance(step, str):
                            print(f"[FLUX FALLBACK] {step}")
                            yield f"Fallback loading: {step}"
                        else:
                            pipeline = step
                            break
                else:
                    pipeline = fallback_result
            else:
                raise
        
        # Debug prompt parameter early
        print(f"[FLUX DEBUG] ===== PROMPT DEBUG =====")
        print(f"[FLUX DEBUG] Received prompt: '{prompt}'")
        print(f"[FLUX DEBUG] Pipeline object: {pipeline}")
        print(f"[FLUX DEBUG] Pipeline type: {type(pipeline)}")
        
        scheduler_name = kwargs.get('scheduler', 'FlowMatchEuler')
        cfg_rescale = kwargs.get('cfg_rescale', 0.0)
        denoise_eta = kwargs.get('denoise_eta', 0.0)
        
        # Apply scheduler if requested
        try:
            from diffusers import EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler
            if scheduler_name == "Euler":
                pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            elif scheduler_name == "FlowMatchEuler":
                # This is usually the default, but set it explicitly to be sure
                pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            else:
                print(f"[FLUX WARNING] Unknown scheduler '{scheduler_name}', using default FlowMatchEulerDiscreteScheduler")
                pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            print(f"[FLUX DEBUG] Scheduler set to {scheduler_name}")
        except Exception as sched_err:
            print(f"[FLUX DEBUG] Scheduler selection failed: {sched_err}")
        
        # Set up generation parameters
        # Build generation kwargs. Only include 'eta' if the user set a non-zero value and
        # avoid passing unsupported arguments to FluxPipeline.__call__.
        generation_kwargs = {
            "prompt": prompt,  # Use the prompt parameter passed to this method
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": torch.Generator().manual_seed(seed),
        }
        # Note: FLUX models don't support eta or guidance_rescale parameters
        
        # Debug: Verify prompt is being passed correctly
        print(f"[FLUX DEBUG] Generation kwargs keys: {list(generation_kwargs.keys())}")
        print(f"[FLUX DEBUG] Prompt value: '{generation_kwargs.get('prompt', 'MISSING')}'")
        print(f"[FLUX DEBUG] Prompt type: {type(generation_kwargs.get('prompt'))}")
        print(f"[FLUX DEBUG] Prompt length: {len(generation_kwargs.get('prompt', '')) if generation_kwargs.get('prompt') else 0}")
        print(f"[FLUX DEBUG] Steps: {generation_kwargs.get('num_inference_steps', 'MISSING')}")
        
        # Validate prompt before calling pipeline
        if not generation_kwargs.get('prompt') or generation_kwargs.get('prompt').strip() == '':
            print(f"[FLUX DEBUG] ❌ Empty or missing prompt detected!")
            print(f"[FLUX DEBUG] Received prompt parameter: '{prompt}'")
            raise ValueError("Prompt parameter is empty or missing")
        
        # Add model-specific parameters
        if "FLUX.1-dev" in model_id:
            generation_kwargs["max_sequence_length"] = max_sequence_length
        
        # Set up step callback for inference progress
        step_progress = [0]  # Mutable for closure
        
        def inference_callback(pipe, step_index, timestep, callback_kwargs):
            step_progress[0] = step_index + 1
            progress_pct = (step_progress[0] / steps) * 100
            print(f"[FLUX INFERENCE] Step {step_progress[0]}/{steps} ({progress_pct:.1f}%)")
            return callback_kwargs
        
        generation_kwargs["callback_on_step_end"] = inference_callback
        
        print(f"[FLUX INFERENCE] 🚀 Starting inference with {steps} steps...")
        print(f"[FLUX INFERENCE] Pipeline ready, calling inference...")
        
        # Run inference directly - accept brief disconnect during inference
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in cast")
            print(f"[FLUX INFERENCE] 🔄 Calling pipeline with {len(generation_kwargs)} parameters...")
            print(f"[FLUX INFERENCE] Pipeline type: {type(pipeline)}")
            print(f"[FLUX INFERENCE] Pipeline available: {pipeline is not None}")
            
            # Double-check prompt is in kwargs
            if 'prompt' not in generation_kwargs:
                print(f"[FLUX INFERENCE] ❌ CRITICAL: 'prompt' key missing from generation_kwargs!")
                raise ValueError("'prompt' key missing from generation_kwargs")
            
            if not generation_kwargs['prompt']:
                print(f"[FLUX INFERENCE] ❌ CRITICAL: prompt value is empty!")
                raise ValueError("prompt value is empty")
            
            print(f"[FLUX INFERENCE] About to call pipeline with prompt: '{generation_kwargs['prompt'][:50]}...'")
            
            # Aggressive memory defragmentation before inference (configurable)
            torch = self._shared_backend.get('torch')
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available() and self._should_do_aggressive_cleanup():
                print(f"[FLUX MEMORY] 🧹 Pre-inference memory cleanup...")
                
                # Check memory before cleanup
                gpu_allocated_before = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved_before = torch.cuda.memory_reserved() / 1024**3
                print(f"[FLUX MEMORY] Before cleanup: {gpu_allocated_before:.2f}GB allocated, {gpu_reserved_before:.2f}GB reserved")
                
                # Aggressive cleanup sequence
                gc.collect()                    # Python garbage collection
                torch.cuda.empty_cache()        # Clear PyTorch cache
                torch.cuda.synchronize()        # Wait for operations
                torch.cuda.empty_cache()        # Clear again after sync
                
                # Try to trigger memory compaction
                try:
                    torch.cuda.memory._dump_snapshot("temp_snapshot.pickle")
                    import os
                    os.remove("temp_snapshot.pickle")
                except:
                    pass  # Snapshot feature might not be available
                
                # Final cleanup
                gc.collect()
                torch.cuda.empty_cache()
                
                # Check memory after cleanup
                gpu_allocated_after = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved_after = torch.cuda.memory_reserved() / 1024**3
                freed_allocated = gpu_allocated_before - gpu_allocated_after
                freed_reserved = gpu_reserved_before - gpu_reserved_after
                
                print(f"[FLUX MEMORY] After cleanup: {gpu_allocated_after:.2f}GB allocated, {gpu_reserved_after:.2f}GB reserved")
                print(f"[FLUX MEMORY] Freed: {freed_allocated:.2f}GB allocated, {freed_reserved:.2f}GB reserved")
                
                # Check fragmentation
                gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
                print(f"[FLUX MEMORY] Available for allocation: {gpu_free:.2f}GB")
                
                if gpu_free < 1.0:  # Less than 1GB free
                    print(f"[FLUX MEMORY] ⚠️ WARNING: Only {gpu_free:.2f}GB free - inference may fail")
                
                print(f"[FLUX INFERENCE] 🧹 Aggressive memory cleanup completed")
            
            # Direct inference call - may briefly disconnect but callback shows progress
            try:
                backend_result = pipeline(**generation_kwargs)
                print(f"[FLUX INFERENCE] ✅ Pipeline call completed!")
            except torch.cuda.OutOfMemoryError as oom_e:
                print(f"[FLUX INFERENCE] ❌ OOM during inference: {oom_e}")
                
                # Emergency VAE fallback (configurable)
                if not self._should_do_emergency_vae_fallback():
                    print(f"[FLUX INFERENCE] Emergency VAE fallback disabled by config - re-raising OOM")
                    raise oom_e
                
                print(f"[FLUX INFERENCE] 🔄 Attempting emergency memory management...")
                
                # Emergency: try to free up more memory by temporarily moving VAE to CPU
                vae_was_on_gpu = False
                original_vae_device = None
                
                try:
                    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                        current_device = next(pipeline.vae.parameters()).device
                        if 'cuda' in str(current_device):
                            print(f"[FLUX EMERGENCY] Moving VAE to CPU temporarily...")
                            original_vae_device = current_device
                            pipeline.vae = pipeline.vae.to('cpu')
                            vae_was_on_gpu = True
                            
                            # Clear cache after moving VAE
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            
                            # Retry inference with VAE on CPU
                            print(f"[FLUX EMERGENCY] Retrying inference with VAE on CPU...")
                            backend_result = pipeline(**generation_kwargs)
                            print(f"[FLUX INFERENCE] ✅ Emergency inference successful!")
                        else:
                            # VAE already on CPU, re-raise the OOM
                            raise oom_e
                    else:
                        # No VAE to move, re-raise the OOM
                        raise oom_e
                        
                finally:
                    # Move VAE back to GPU if we moved it
                    if vae_was_on_gpu and original_vae_device is not None:
                        try:
                            print(f"[FLUX EMERGENCY] Moving VAE back to {original_vae_device}...")
                            pipeline.vae = pipeline.vae.to(original_vae_device)
                        except Exception as restore_e:
                            print(f"[FLUX EMERGENCY] ⚠️ Could not restore VAE to GPU: {restore_e}")
            except Exception as e:
                print(f"[FLUX INFERENCE] ❌ Non-OOM error during inference: {e}")
                raise
            except RuntimeError as e:
                # Catch missing kernel errors from bitsandbytes after forward pass starts
                if "no kernel image" in str(e).lower() and quantization in ["4-bit", "8-bit"]:
                    # On Hopper GPUs the 8-bit kernels may be missing – retry with 4-bit to stay quantized
                    print("[FLUX FALLBACK] Kernel error – retrying in 4-bit …")
                    quantization_retry = "4-bit"
                    pipeline_fp = self._pipeline_loader.get_pipeline_with_progress(model_id, quantization_retry, system_constraints)
                    backend_result = pipeline_fp(**generation_kwargs)
                    quantization = quantization_retry  # update for logs
                else:
                    raise
            print(f"[FLUX INFERENCE] 🖼️ Extracting image from result...")
        
        print(f"[FLUX DEBUG] Backend result type after all processing: {type(backend_result)}")
        
        # Handle FluxPipelineOutput from diffusers and return tuple
        if hasattr(backend_result, 'images') and backend_result.images:
            generated_image = backend_result.images[0]  # First image  
            generation_info = {
                "backend": "Diffusers (CUDA)",
                "model": model_id,
                "actual_seed": seed,
                "steps": steps,
                "guidance": guidance,
                "width": width,
                "height": height
            }
            print(f"[FLUX DEBUG] Extracted from FluxPipelineOutput: image={type(generated_image)}")
            print(f"[FLUX DEBUG] Returning tuple: ({type(generated_image)}, {type(generation_info)})")
            
            # Release working memory reservation
            if working_memory_reserved:
                if kwargs.get('width', 512) * kwargs.get('height', 512) > 1024 * 1024:
                    self._memory_manager.release_working_memory(OperationType.LONG_CONTEXT)
                else:
                    self._memory_manager.release_working_memory(OperationType.INFERENCE)
            
            # Print performance stats for this generation
            self._memory_manager.print_performance_stats()
            
            yield (generated_image, generation_info)
    
    def _should_clear_gpu_cache(self) -> bool:
        """Smart check to determine if GPU cache clearing is needed"""
        torch = self._shared_backend.get('torch')
        if not torch or not hasattr(torch, 'cuda') or not torch.cuda.is_available():
            return False
            
        # On first run, GPU cache is empty - no need to clear
        if self._first_run:
            print(f"[FLUX MEMORY] First run detected - GPU cache is clean")
            return False
        
        # torch is already validated above
        # torch = self._shared_backend.get('torch')
        # if not torch:
        #     return False
        
        # Check current GPU memory usage
        device = torch.cuda.current_device()
        allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
        reserved_mb = torch.cuda.memory_reserved(device) / 1024**2
        
        # If significant memory is allocated (>500MB), clear cache
        if allocated_mb > 500:
            print(f"[FLUX MEMORY] GPU memory in use: {allocated_mb:.1f}MB allocated, {reserved_mb:.1f}MB reserved - clearing cache")
            return True
        
        # If reserved memory is high but allocated is low, we have fragmentation
        if reserved_mb > 1000 and allocated_mb < reserved_mb * 0.5:
            print(f"[FLUX MEMORY] GPU fragmentation detected: {allocated_mb:.1f}MB allocated vs {reserved_mb:.1f}MB reserved - clearing cache")
            return True
            
        print(f"[FLUX MEMORY] GPU memory is clean: {allocated_mb:.1f}MB allocated, {reserved_mb:.1f}MB reserved - skipping cache clear")
        return False