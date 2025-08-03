from __future__ import annotations
import os, sys, subprocess, warnings
from typing import Any

def get_cuda_backend():
    return DiffusersFluxBackend()

def _ensure_bitsandbytes():
    """Dynamically install a BnB wheel that contains kernels for the current GPU.

    • Consumer GPUs (≤ sm_86) – the default PyPI wheel works
    • Hopper (sm_90) / future cards – pull pre-release wheel with sm_90 kernels
    The function is a no-op on CPU / Apple Silicon.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return  # CPU / MPS – nothing to do
        # Only import here so that the try/except below is meaningful
        import importlib, contextlib
        def _try_import_ok():
            try:
                bnb = importlib.import_module("bitsandbytes")
                # quick functional smoke-test on Hopper
                major, _minor = torch.cuda.get_device_capability(0)
                if major < 9:
                    return True
                t = torch.ones(4, device="cuda")
                with contextlib.suppress(RuntimeError):
                    bnb.functional.quantize_4bit(t, quant_type="nf4")[0]
                    return True
                return False
            except (OSError, RuntimeError, ModuleNotFoundError):
                return False
        if _try_import_ok():
            return  # wheel already usable

        major, _ = torch.cuda.get_device_capability(0)
        # Prefer HuggingFace CUDA-12.4 wheels for Ada (sm_8x) and Hopper (≥sm_90)
        wheel_spec = (
            "bitsandbytes>=0.46.1 --extra-index-url https://huggingface.github.io/bitsandbytes-wheels/cu124"
            if major >= 8 else
            "bitsandbytes>=0.46.0"
        )
        print(f"[FLUX SETUP] Installing compatible bitsandbytes wheel: {wheel_spec}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + wheel_spec.split()
        )
        importlib.import_module("bitsandbytes")
    except Exception as e:
        # Don't hard-fail – _get_pipeline fallback will handle full precision
        print(f"[FLUX SETUP] bitsandbytes installation failed: {e}")

class DiffusersFluxBackend(FluxBackend):
    """Diffusers-based backend for CUDA/CPU using shared backend"""
    
    def __init__(self):
        """Initialize the FluxInference node with optimized GPU memory management"""
        super().__init__()
        
        # Configure PyTorch CUDA allocator via environment variables (safe before torch import)
        import os
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
                            'garbage_collection_threshold:0.8,max_split_size_mb:128,expandable_segments:True')
        
        print(f"[FLUX INIT] 🔧 Configured CUDA allocator for improved memory management")
        print(f"[FLUX INIT] 🔧 CUDA allocator config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
        print(f"[FLUX INIT] 💡 Advanced users: Set PYTORCH_CUDA_ALLOC_CONF env var to customize memory management")
        
        # Ensure bitsandbytes kernels are present (installs wheel if needed)
        _ensure_bitsandbytes()
        
        # Initialize pipeline cache for model reuse
        self._pipeline_cache = {}
        
        # Initialize shared backend for heavy library imports
        self._shared_backend = get_shared_backend()
        
        # Initialize memory management flags
        self._last_cleanup_failed = False
        self._cleanup_failed_memory = 0.0

    @staticmethod
    def _normalize_memory_dict(mem: dict, gpu_device: int | str = 0) -> dict:
        """Return a copy of *mem* with all GPU keys as 'cuda:N' strings.
        Accepts variants: int, numeric str, 'auto'. Raises on unknown keys."""
        if not mem:
            return {}
        norm = {}
        for key, val in mem.items():
            if key == 'cpu':
                norm['cpu'] = val
            elif key == 'auto':
                norm[f'cuda:{gpu_device}'] = val
            elif isinstance(key, int):
                norm[key] = val  # keep integers as-is (accelerate prefers ints)
            elif isinstance(key, str) and key.isdigit():
                # numeric string → int
                norm[int(key)] = val
            elif isinstance(key, str) and key.startswith('cuda:'):
                norm[key] = val
            else:
                raise ValueError(
                    f"Invalid device key in max_memory dict: {key!r}. Use 'cpu', 'auto', or GPU indices like 0 / 'cuda:0'."
                )
        return norm
    
    def is_available(self) -> bool:
        """Check if backend is available using pre-loaded modules"""
        if not self._shared_backend['available']:
            print(f"[FLUX BACKEND] Shared backend not available: {self._shared_backend.get('error', 'Unknown error')}")
            return False
        
        # Quick availability check using pre-loaded modules
        try:
            torch = self._shared_backend['torch']
            print(f"[FLUX BACKEND] Using pre-loaded PyTorch, CUDA available: {torch.cuda.is_available()}")
            return True
        except Exception as e:
            print(f"[FLUX BACKEND] Error checking availability: {e}")
            return False
    
    def get_name(self) -> str:
        """Get backend name using pre-loaded modules"""
        try:
            torch = self._shared_backend['torch']
            if torch.cuda.is_available():
                return "Diffusers (CUDA)"
            else:
                return "Diffusers (CPU)"
        except Exception:
            return "Diffusers"
    
    def validate_model_id(self, model_id: str) -> bool:
        return "FLUX.1-dev" in model_id or "FLUX.1-schnell" in model_id
    
    def _check_memory_safety(self, quantization: str, system_constraints: dict = None):
        """Check if system has enough memory to safely load FLUX model."""
        torch = self._shared_backend['torch']
        
        # Get available memory info
        if torch.cuda.is_available():
            gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
            
            # Handle 'auto' device selection - convert to actual device number
            if gpu_device == 'auto':
                gpu_device = 0  # Default to first GPU when auto is selected
            
            gpu_props = torch.cuda.get_device_properties(gpu_device)
            total_gpu_memory_gb = gpu_props.total_memory / 1024**3
            allocated_gpu_memory_gb = torch.cuda.memory_allocated(gpu_device) / 1024**3
            available_gpu_memory_gb = total_gpu_memory_gb - allocated_gpu_memory_gb
            
            print(f"[FLUX MEMORY] GPU {gpu_device}: {available_gpu_memory_gb:.1f}GB available / {total_gpu_memory_gb:.1f}GB total")
            
            # Get memory limits from system constraints or use sensible defaults
            default_memory_requirements = {
                "4-bit": 8.0,    # Default minimum for 4-bit quantization
                "8-bit": 12.0,   # Default minimum for 8-bit quantization  
                "none": 24.0     # Default minimum for full precision
            }
            
            # Use system constraints for memory limits if provided
            if system_constraints:
                required_memory = system_constraints.get(f'{quantization}_memory_gb', 
                                                        system_constraints.get('gpu_memory_gb',
                                                                             default_memory_requirements.get(quantization, 24.0)))
                print(f"[FLUX MEMORY] Using system constraint memory limit: {required_memory}GB for {quantization}")
            else:
                required_memory = default_memory_requirements.get(quantization, 24.0)
                print(f"[FLUX MEMORY] Using default memory requirement: {required_memory}GB for {quantization}")
            
            # Check if constraint is too restrictive
            min_viable_memory = 6.0  # Absolute minimum to attempt loading
            if required_memory < min_viable_memory:
                print(f"[FLUX MEMORY] ⚠️ Memory constraint ({required_memory}GB) is very low - may cause failures")
            
            if available_gpu_memory_gb < required_memory:
                error_msg = (
                    f"⚠️ MEMORY WARNING: Only {available_gpu_memory_gb:.1f}GB available, "
                    f"but {quantization} quantization limit is {required_memory}GB. "
                    f"This may cause system hangs or OOM errors!"
                )
                print(f"[FLUX MEMORY] {error_msg}")
                
                # Only raise error if user hasn't explicitly set a constraint allowing this
                user_override = system_constraints and (
                    'allow_low_memory' in system_constraints or 
                    'gpu_memory_gb' in system_constraints or
                    f'{quantization}_memory_gb' in system_constraints
                )
                
                if not user_override and available_gpu_memory_gb < min_viable_memory:
                    # Check if this is likely a quantized model cleanup failure and we should wait for cycles
                    if (hasattr(self, '_pipeline_cache') and len(self._pipeline_cache) == 0 and 
                        available_gpu_memory_gb < 6.0 and hasattr(self, '_shared_backend')):
                        
                        # Return a special marker indicating async memory check needed
                        return "ASYNC_MEMORY_CHECK_NEEDED"
                    
                    # Prepare error message with helpful hints
                    cleanup_failed_hint = ""
                    auto_fallback_suggestion = ""
                    
                    if hasattr(self, '_pipeline_cache') and len(self._pipeline_cache) == 0 and available_gpu_memory_gb < 6.0:
                        cleanup_failed_hint = (
                            "\n💡 This appears to be a quantized model cleanup failure. "
                            "Quantized models can be 'sticky' in GPU memory. "
                            "\n💡 SOLUTIONS: "
                            "\n   1. Set 'allow_low_memory': true to override this safety check"
                            "\n   2. Use 'none' (full precision) instead of quantization for this model"
                            "\n   3. Restart the workflow to fully clear GPU memory"
                        )
                        
                        # Smart fallback suggestions based on available memory
                        if quantization == "4-bit":
                            if available_gpu_memory_gb >= 10.0:
                                auto_fallback_suggestion = "\n🔄 SMART SUGGESTION: Try 8-bit quantization instead (you have enough memory)"
                            elif available_gpu_memory_gb >= 20.0:
                                auto_fallback_suggestion = "\n🔄 SMART SUGGESTION: Try 'none' (full precision) instead"
                        elif quantization == "8-bit":
                            if available_gpu_memory_gb >= 20.0:
                                auto_fallback_suggestion = "\n🔄 SMART SUGGESTION: Try 'none' (full precision) instead"
                        
                        if not auto_fallback_suggestion and available_gpu_memory_gb < 6.0:
                            auto_fallback_suggestion = "\n🔄 RECOMMENDATION: Restart workflow to fully clear GPU memory (current memory too low)"
                    
                    raise RuntimeError(
                        f"Insufficient GPU memory: {available_gpu_memory_gb:.1f}GB available, "
                        f"below minimum viable ({min_viable_memory}GB). Set 'allow_low_memory': true "
                        f"in system constraints to override this safety check.{cleanup_failed_hint}{auto_fallback_suggestion}"
                    )
                elif user_override:
                    print(f"[FLUX MEMORY] ⚠️ User override detected - proceeding despite low memory warning")
        else:
            print(f"[FLUX MEMORY] CUDA not available - using CPU (will be very slow)")
            
        # Check system RAM for CPU fallback
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / 1024**3
            print(f"[FLUX MEMORY] System RAM: {available_ram_gb:.1f}GB available")
            
            if available_ram_gb < 16.0:
                print(f"[FLUX MEMORY] ⚠️ Low system RAM ({available_ram_gb:.1f}GB), may cause swapping")
        except ImportError:
            print(f"[FLUX MEMORY] psutil not available, cannot check system RAM")

    def _manage_pipeline_cache(self, new_cache_key: str, quantization: str, system_constraints: dict = None):
        """Intelligently manage pipeline cache to prevent memory issues when switching quantization modes
        
        Cache key format: '{model_id}_{quantization}' (e.g., 'FLUX.1-dev_8-bit')
        - Same model + same quantization = cache hit, no clearing ✅
        - Different model OR different quantization = cache miss, check if clearing needed
        """
        if new_cache_key in self._pipeline_cache:
            return
        if not self._pipeline_cache:
            return
        
        # Get torch from shared backend - this is where torch is actually loaded
        torch = self._shared_backend['torch']
        
        # Log CUDA allocator configuration now that torch is available
        if not hasattr(self, '_cuda_allocator_logged'):
            try:
                if torch.cuda.is_available():
                    print(f"[FLUX CACHE] ✅ PyTorch CUDA allocator active with config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
                self._cuda_allocator_logged = True
            except Exception as e:
                print(f"[FLUX CACHE] ⚠️ Could not verify CUDA allocator config: {e}")
                self._cuda_allocator_logged = True
        
        # Check available GPU memory before deciding to clear
        gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
        if isinstance(gpu_device, str) and gpu_device == 'auto':
            gpu_device = 0
        
        available_gpu_memory_gb = (torch.cuda.get_device_properties(gpu_device).total_memory - 
                                   torch.cuda.memory_allocated(gpu_device)) / 1024**3
        total_gpu_memory_gb = torch.cuda.get_device_properties(gpu_device).total_memory / 1024**3
        
        # Estimate memory needed for different quantization modes  
        if quantization == "4-bit":
            needed_memory = 8.0  # Conservative estimate for 4-bit + overhead
        elif quantization == "8-bit":
            needed_memory = 12.0  # Conservative estimate for 8-bit + overhead  
        else:  # none/full precision
            needed_memory = 20.0  # Conservative estimate for full precision
        
        print(f"[FLUX CACHE] Available GPU memory: {available_gpu_memory_gb:.1f}GB")
        print(f"[FLUX CACHE] Estimated need for {quantization}: {needed_memory}GB")
        
        if available_gpu_memory_gb >= needed_memory:
            print(f"[FLUX CACHE] ✅ Sufficient memory available for {quantization} quantization")
            return
        
        # Advanced cleanup for quantized models
        if available_gpu_memory_gb < needed_memory:
            print(f"[FLUX CACHE] 🧹 Insufficient memory for {quantization} quantization")
            
            # Step 1: Enable PyTorch reference cycle detection for debugging  
            try:
                from torch.utils.viz._cycles import observe_tensor_cycles
                print(f"[FLUX CACHE] 🔍 Enabling reference cycle detection for cleanup debugging")
                
                def cycle_callback(html):
                    print(f"[FLUX CACHE] ⚠️ Reference cycle detected during cleanup - this may explain sticky memory")
                
                observe_tensor_cycles(cycle_callback)
            except ImportError:
                print(f"[FLUX CACHE] ⚠️ Reference cycle detection not available in this PyTorch version")
            
            # Step 2: Advanced pipeline cleanup
            old_pipelines = list(self._pipeline_cache.values())
            old_cache_keys = list(self._pipeline_cache.keys())
            print(f"[FLUX CACHE] 🧹 Clearing cached models: {old_cache_keys}")
            
            for old_pipeline in old_pipelines:
                try:
                    # Reset device mapping before any operations
                    if hasattr(old_pipeline, 'reset_device_map'):
                        print(f"[FLUX CACHE] 🔧 Resetting device map for pipeline...")
                        old_pipeline.reset_device_map()
                    
                    # For quantized models, don't try to move to CPU (bitsandbytes limitation)
                    # Instead, explicitly delete components
                    print(f"[FLUX CACHE] 🗑️ Explicitly clearing pipeline components...")
                    components_cleared = 0
                    for attr_name in ['transformer', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2']:
                        if hasattr(old_pipeline, attr_name):
                            try:
                                component = getattr(old_pipeline, attr_name)
                                # Clear component parameters if it has them
                                if hasattr(component, 'parameters'):
                                    for param in component.parameters():
                                        if param.data is not None:
                                            del param.data
                                        if param.grad is not None:
                                            del param.grad
                                del component
                                delattr(old_pipeline, attr_name)
                                components_cleared += 1
                            except Exception as e:
                                print(f"[FLUX CACHE] Warning clearing {attr_name}: {e}")
                    print(f"[FLUX CACHE] 🗑️ Cleared {components_cleared} pipeline components")
                    
                    # Clear any remaining attributes
                    for attr in dir(old_pipeline):
                        if not attr.startswith('_') and hasattr(old_pipeline, attr):
                            try:
                                attr_val = getattr(old_pipeline, attr)
                                if hasattr(attr_val, 'cuda') or str(type(attr_val)).find('torch') != -1:
                                    delattr(old_pipeline, attr)
                            except:
                                pass
                    
                except Exception as e:
                    print(f"[FLUX CACHE] Warning during pipeline cleanup: {e}")
            
            # Step 3: Clear cache and explicitly delete references
            self._pipeline_cache.clear()
            del old_pipelines
            
            # Step 4: Force synchronization before cleanup
            print(f"[FLUX CACHE] 🔄 Forcing CUDA synchronization...")
            torch.cuda.synchronize()
            
            # Step 5: Multi-round aggressive GPU memory cleanup for quantized models
            print(f"[FLUX CACHE] 🧹 Starting aggressive GPU memory cleanup for quantized models...")
            import gc
            import time
            
            for cleanup_round in range(7):  # Increased rounds
                # Multiple different cleanup approaches
                if cleanup_round == 0:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: Standard cleanup")
                    torch.cuda.empty_cache()
                elif cleanup_round == 1:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: Python garbage collection")
                    gc.collect()
                elif cleanup_round == 2:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: CUDA + Python GC")
                    torch.cuda.empty_cache()
                    gc.collect()
                elif cleanup_round == 3:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: CUDA synchronize + cleanup")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                elif cleanup_round == 4:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: IPC collection")
                    try:
                        torch.cuda.ipc_collect()
                    except:
                        pass
                    torch.cuda.empty_cache()
                elif cleanup_round == 5:
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: Reset memory stats")
                    try:
                        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                            torch.cuda.reset_peak_memory_stats()
                        if hasattr(torch.cuda, 'reset_max_memory_allocated'):
                            torch.cuda.reset_max_memory_allocated()
                    except:
                        pass
                    torch.cuda.empty_cache()
                else:  # Round 6 - final aggressive round
                    print(f"[FLUX CACHE] Round {cleanup_round + 1}: Combined aggressive cleanup")
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except:
                        pass
                
                current_allocated = torch.cuda.memory_allocated(gpu_device) / 1024**3
                print(f"[FLUX CACHE] Cleanup round {cleanup_round + 1}: {current_allocated:.1f}GB still allocated")
                
                # Note: Removed blocking delays - cleanup runs at full speed now
            
            # Step 6: Check memory improvement
            new_available = (torch.cuda.get_device_properties(gpu_device).total_memory - 
                           torch.cuda.memory_allocated(gpu_device)) / 1024**3
            freed_memory = new_available - available_gpu_memory_gb
            print(f"[FLUX CACHE] ✅ Freed {freed_memory:.1f}GB, now have {new_available:.1f}GB available")
            
            # Step 7: Nuclear option if memory is still insufficient
            final_available = new_available  # Initialize in broader scope
            if new_available < needed_memory:
                print(f"[FLUX CACHE] ⚠️ Still insufficient memory after cleanup ({new_available:.1f}GB < {needed_memory}GB)")
                print(f"[FLUX CACHE] 🚨 Quantized models can be 'sticky' in GPU memory - trying nuclear cleanup...")
                
                try:
                    if torch.cuda.is_available():
                        print(f"[FLUX CACHE] 💣 Attempting CUDA context reset to force memory cleanup...")
                        
                        # Advanced CUDA memory reset techniques
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                        
                        # Reset all memory statistics if available
                        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                            torch.cuda.reset_accumulated_memory_stats()
                        if hasattr(torch.cuda, 'reset_max_memory_allocated'):
                            torch.cuda.reset_max_memory_allocated()
                        if hasattr(torch.cuda, 'reset_max_memory_cached'):
                            torch.cuda.reset_max_memory_cached()
                        
                        # One final aggressive cleanup
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        final_available = total_gpu_memory_gb - torch.cuda.memory_allocated(gpu_device) / 1024**3
                        nuclear_freed = final_available - new_available
                        print(f"[FLUX CACHE] 💣 Nuclear cleanup freed additional {nuclear_freed:.1f}GB")
                        print(f"[FLUX CACHE] 💣 Final available: {final_available:.1f}GB")
                        
                        # Wait for reference cycle detection to complete (it runs asynchronously)
                        print(f"[FLUX CACHE] ⏱️ Waiting for reference cycle cleanup to complete...")
                        # Note: This wait is now non-blocking and handled by memory safety check
                        
                        # Check memory again after waiting for cycles to clear
                        torch.cuda.synchronize()
                        post_cycle_available = total_gpu_memory_gb - torch.cuda.memory_allocated(gpu_device) / 1024**3
                        cycle_freed = post_cycle_available - final_available
                        
                        if cycle_freed > 0.1:  # If significant memory was freed by cycle detection
                            print(f"[FLUX CACHE] ✅ Reference cycle cleanup freed additional {cycle_freed:.1f}GB!")
                            print(f"[FLUX CACHE] ✅ Final memory after cycle cleanup: {post_cycle_available:.1f}GB")
                            final_available = post_cycle_available
                        
                        if final_available >= needed_memory:
                            print(f"[FLUX CACHE] ✅ Cleanup succeeded! Enough memory now available")
                            return  # Exit early if we now have enough memory
                        else:
                            print(f"[FLUX CACHE] ⚠️ Even with cycle cleanup insufficient - this is expected behavior")
                            print(f"[FLUX CACHE] 💡 Quantized models can require restart to fully clear GPU memory")
                    
                except Exception as e:
                    print(f"[FLUX CACHE] ⚠️ Nuclear cleanup failed: {e}")
                    # If nuclear cleanup failed, keep using the best available memory value
                    final_available = new_available
                    
                print(f"[FLUX CACHE] ⚠️ Memory safety check will still trigger if insufficient memory remains")
                print(f"[FLUX CACHE] 💡 WORKAROUND: Set 'allow_low_memory': true in system constraints to force load")
                print(f"[FLUX CACHE] 💡 ALTERNATIVE: Restart the workflow to fully clear quantized model memory")
                
                # Smart fallback: Suggest automatic quantization adjustment (use final_available which includes cycle cleanup)
                if quantization == "4-bit" and final_available >= 10.0:
                    print(f"[FLUX CACHE] 🔄 AUTO-FALLBACK OPTION: You have {final_available:.1f}GB - try 8-bit instead")
                elif quantization in ["4-bit", "8-bit"] and final_available >= 20.0:
                    print(f"[FLUX CACHE] 🔄 AUTO-FALLBACK OPTION: You have {final_available:.1f}GB - try 'none' (full precision)")
                elif final_available < 6.0:
                    print(f"[FLUX CACHE] 🔄 RECOMMENDED: Restart workflow to fully clear GPU memory ({final_available:.1f}GB too low for any model)")
                
                # Store the failed cleanup info for smarter error handling
                if hasattr(self, '_pipeline_cache'):
                    self._last_cleanup_failed = True
                    self._cleanup_failed_memory = final_available
                        
        else:
            print(f"[FLUX CACHE] ✅ Sufficient memory available for {quantization} quantization")

    def _test_quantization_setup(self, quantization: str) -> bool:
        """Test quantization setup without loading models"""
        print(f"[FLUX DEBUG] Testing {quantization} quantization setup...")
        
        try:
            if not self._shared_backend['available']:
                raise ImportError(f"Shared backend not available: {self._shared_backend.get('error', 'Unknown error')}")
            
            BitsAndBytesConfig = self._shared_backend['BitsAndBytesConfig']
            torch = self._shared_backend['torch']
            
            print(f"[FLUX DEBUG] BitsAndBytesConfig available, testing {quantization} configuration...")
            
            if quantization == "4-bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    load_in_8bit_fp32_cpu_offload=True  # Enable CPU offload for 4-bit as well
                )
                print(f"[FLUX DEBUG] ✅ Created 4-bit BitsAndBytesConfig with CPU offload")
            elif quantization == "8-bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    load_in_8bit_fp32_cpu_offload=True  # Enable CPU offload for insufficient GPU memory
                )
                print(f"[FLUX DEBUG] ✅ Created 8-bit BitsAndBytesConfig with CPU offload")
            else:
                raise ValueError(f"Unsupported quantization: {quantization}")
            
            print(f"[FLUX DEBUG] ✅ {quantization} quantization configuration test successful")
            return True
            
        except Exception as e:
            print(f"[FLUX DEBUG] ❌ {quantization} quantization test failed: {e}")
            raise
    
    def _get_pipeline_with_progress(self, model_id: str, quantization: str = "none", system_constraints: dict = None):
        """Load pipeline directly (no generator)"""
        print(f"[FLUX DEBUG] ===== _get_pipeline_with_progress CALLED =====")
        print(f"[FLUX DEBUG] model_id: {model_id}")
        print(f"[FLUX DEBUG] quantization: '{quantization}' (type: {type(quantization)})")
        print(f"[FLUX DEBUG] system_constraints: {system_constraints}")
        # Fail-fast validation of max_memory dict
        if system_constraints:
            system_constraints['max_memory'] = self._normalize_memory_dict(system_constraints.get('max_memory', {}), gpu_device=system_constraints.get('gpu_device', 0))
        
        cache_key = f"{model_id}_{quantization}"
        print(f"[FLUX DEBUG] cache_key: {cache_key}")
        
        # Check cache first
        if cache_key in self._pipeline_cache:
            print(f"[FLUX LOADING] ✅ Using cached pipeline for {cache_key}")
            return self._pipeline_cache[cache_key]
        
        # Intelligent cache management: check if we need to clear old models
        self._manage_pipeline_cache(cache_key, quantization, system_constraints)
        
        # Memory safety check before loading
        memory_check_result = self._check_memory_safety(quantization, system_constraints)
        if memory_check_result == "ASYNC_MEMORY_CHECK_NEEDED":
            return "ASYNC_MEMORY_CHECK_NEEDED"
        
        # Use pre-loaded modules from shared backend
        if not self._shared_backend['available']:
            raise ImportError(f"Shared backend not available: {self._shared_backend.get('error', 'Unknown error')}")
        
        FluxPipeline = self._shared_backend['FluxPipeline']
        torch = self._shared_backend['torch']
        
        # Debug quantization parameter early
        print(f"[FLUX DEBUG] ===== QUANTIZATION DEBUG START =====")
        print(f"[FLUX DEBUG] Raw quantization parameter: '{quantization}' (type: {type(quantization)})")
        print(f"[FLUX DEBUG] Quantization in ['4-bit', '8-bit']: {quantization in ['4-bit', '8-bit']}")
        print(f"[FLUX DEBUG] Model ID: {model_id}")
        print(f"[FLUX DEBUG] System constraints: {system_constraints}")
        
        if quantization in ["4-bit", "8-bit"]:
            print(f"[FLUX DEBUG] ===== ENTERING QUANTIZATION SETUP =====")
        else:
            print(f"[FLUX DEBUG] ===== SKIPPING QUANTIZATION (using full precision) =====")
            print(f"[FLUX DEBUG] Quantization value that caused skip: '{quantization}'")
        
        # Setup quantization config if needed using shared backend
        quantization_config = None
        bnb_config = None
        
        if quantization in ["4-bit", "8-bit"]:
            print(f"[FLUX DEBUG] Setting up {quantization} quantization...")
            print(f"[FLUX DEBUG] Shared backend available: {self._shared_backend.get('available', False)}")
            print(f"[FLUX DEBUG] Shared backend keys: {list(self._shared_backend.keys())}")
            
            try:
                print(f"[FLUX DEBUG] Attempting to get BitsAndBytesConfig from shared backend...")
                BitsAndBytesConfig = self._shared_backend['BitsAndBytesConfig']
                print(f"[FLUX DEBUG] ✅ BitsAndBytesConfig retrieved: {BitsAndBytesConfig}")
                print(f"[FLUX DEBUG] BitsAndBytesConfig available, setting up {quantization} quantization...")
                
                if quantization == "4-bit":
                    print(f"[FLUX DEBUG] Creating 4-bit BitsAndBytesConfig...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    print(f"[FLUX DEBUG] ✅ 4-bit config created: {bnb_config}")
                elif quantization == "8-bit":
                    print(f"[FLUX DEBUG] Creating 8-bit BitsAndBytesConfig...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        load_in_8bit_fp32_cpu_offload=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                        llm_int8_skip_modules=None,
                        llm_int8_threshold=6.0
                    )
                    print(f"[FLUX DEBUG] ✅ 8-bit config created: {bnb_config}")
                
                quantization_config = bnb_config
                print(f"[FLUX DEBUG] ✅ Final quantization_config set: {quantization_config}")
                print(f"[FLUX DEBUG] ✅ Using direct BitsAndBytesConfig for {quantization} quantization")
                    
            except KeyError as ke:
                print(f"[FLUX ERROR] ❌ KeyError accessing shared backend: {ke}")
                print(f"[FLUX ERROR] Available shared backend keys: {list(self._shared_backend.keys())}")
                quantization_config = None
                bnb_config = None
            except Exception as e:
                print(f"[FLUX ERROR] ❌ Quantization setup failed: {e}")
                print(f"[FLUX ERROR] Exception type: {type(e)}")
                import traceback
                print(f"[FLUX ERROR] Traceback: {traceback.format_exc()}")
                quantization_config = None
                bnb_config = None
        else:
            print(f"[FLUX DEBUG] No quantization requested (quantization='{quantization}')")
        
        # Load pipeline with quantization and progress callbacks
        pipeline = None
        
        print(f"[FLUX DEBUG] About to check quantization_config: {quantization_config}")
        print(f"[FLUX DEBUG] Quantization requested: '{quantization}'")
        print(f"[FLUX DEBUG] quantization_config is None: {quantization_config is None}")
        
        if quantization_config is not None:
            try:
                print(f"[FLUX LOADING] 🔄 Loading {model_id} with component-level quantization ({quantization})...")
                
                from diffusers import FluxTransformer2DModel
                
                # Set up loading parameters
                loading_kwargs = {
                    "subfolder": "transformer",
                    "quantization_config": quantization_config,
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                }
                
                # Add memory constraints for quantization
                if quantization in ["4-bit", "8-bit"]:
                    gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                    if gpu_device == 'auto':
                        gpu_device = 0
                    
                    if system_constraints:
                        max_memory = system_constraints.get('max_memory', {})
                        if max_memory:
                            fixed_memory = {}
                            for key, value in max_memory.items():
                                if key == 'auto':
                                    fixed_memory[gpu_device] = value
                                elif isinstance(key, int):
                                    fixed_memory[key] = value
                                elif isinstance(key, str) and key.isdigit():
                                    fixed_memory[int(key)] = value
                                elif isinstance(key, str) and key.startswith("cuda:"):
                                    fixed_memory[key] = value
                                else:
                                    fixed_memory[key] = value
                            loading_kwargs["max_memory"] = fixed_memory
                            # Remove device_map when explicit memory dict is provided to avoid key casting
                            loading_kwargs.pop("device_map", None)
                    else:
                        # Don't artificially limit GPU memory - let it use what's available
                        print(f"[FLUX LOADING] Using default memory allocation (no artificial limits)")
                
                # Direct transformer loading (no more yielding)
                print(f"[FLUX LOADING] Loading transformer directly...")
                transformer = FluxTransformer2DModel.from_pretrained(model_id, **loading_kwargs)
                print(f"[FLUX LOADING] Transformer loading completed!")
                
                # Load pipeline with quantized transformer
                pipeline_kwargs = {
                    "transformer": transformer,
                    "torch_dtype": torch.bfloat16,
                    "use_safetensors": True,
                }
                
                if quantization in ["4-bit", "8-bit"] and torch.cuda.is_available():
                    gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                    max_memory = system_constraints.get('max_memory', {}) if system_constraints else {}
                    
                    if max_memory:
                        modified_memory = {}
                        for key, value in max_memory.items():
                            if key == 'auto':
                                modified_memory[gpu_device] = value
                            elif isinstance(key, int):
                                modified_memory[key] = value
                            elif isinstance(key, str) and key.isdigit():
                                modified_memory[int(key)] = value
                            elif isinstance(key, str) and key.startswith("cuda:"):
                                modified_memory[key] = value
                            elif key == 'cpu':
                                modified_memory['cpu'] = "8GB"
                            else:
                                modified_memory[key] = value
                        pipeline_kwargs.update({
                            "device_map": "balanced",
                            "max_memory": modified_memory,
                            "low_cpu_mem_usage": True
                        })
                        # Remove device_map since explicit max_memory is set
                        pipeline_kwargs.pop("device_map", None)
                    else:
                        if quantization == "4-bit":
                            memory_config = {gpu_device: "6GB", "cpu": "8GB"}
                        else:
                            memory_config = {gpu_device: "10GB", "cpu": "8GB"}
                        
                        pipeline_kwargs.update({
                            "device_map": "balanced",
                            "max_memory": memory_config,
                            "low_cpu_mem_usage": True
                        })
                        pipeline_kwargs.pop("device_map", None)
                
                # Direct pipeline loading (no more yielding)
                print(f"[FLUX LOADING] Loading pipeline directly...")
                pipeline = FluxPipeline.from_pretrained(model_id, **pipeline_kwargs)
                print(f"[FLUX LOADING] Pipeline assembly completed!")
                
                # Device placement optimization (existing code)
                if torch.cuda.is_available():
                    gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                    device_str = "cuda" if str(gpu_device) in ("auto", "cuda") else f"cuda:{gpu_device}"
                    
                    if quantization in ["4-bit", "8-bit"]:
                        print(f"[FLUX LOADING] 🔧 Checking VAE placement for {quantization} quantization...")
                        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                            vae_device = next(pipeline.vae.parameters()).device
                            if 'cpu' in str(vae_device):
                                try:
                                    pipeline.vae = pipeline.vae.to(device_str)
                                    print(f"[FLUX LOADING] ✅ VAE moved to {device_str}")
                                    # For 8-bit we must keep everything on the same device; for 4-bit we leave encoders on CPU
                                    if quantization == "8-bit":
                                        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                                            pipeline.text_encoder = pipeline.text_encoder.to(device_str)
                                            print(f"[FLUX LOADING] ✅ CLIP moved to {device_str}")
                                        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
                                            pipeline.text_encoder_2 = pipeline.text_encoder_2.to(device_str)
                                            print(f"[FLUX LOADING] ✅ T5 moved to {device_str}")
                                    else:
                                        print(f"[FLUX LOADING] 🧠 Keeping text encoders on CPU for 4-bit to save VRAM")
                                except Exception as e:
                                    print(f"[FLUX LOADING] ❌ Could not move VAE: {e}")
                
                print(f"[FLUX LOADING] ✅ Successfully loaded with component-level quantization")
                
            except Exception as e1:
                print(f"[FLUX LOADING] ❌ Component-level quantization failed: {e1}")
                print(f"[FLUX DEBUG] Quantization exception type: {type(e1)}")
                import traceback
                print(f"[FLUX DEBUG] Quantization traceback: {traceback.format_exc()}")
                pipeline = None
        
        if pipeline is None:
            print(f"[FLUX LOADING] ⚠️ Loading {model_id} without quantization (FULL PRECISION)")
            print(f"[FLUX DEBUG] Fallback reason: quantization_config was {quantization_config}, quantization was '{quantization}'")
            if quantization in ["4-bit", "8-bit"]:
                print(f"[FLUX DEBUG] Quantization was requested but failed - check error logs above")
            
            # Direct full precision loading (no more yielding)
            print(f"[FLUX LOADING] Loading full precision model directly...")
            pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            print(f"[FLUX LOADING] Full precision model loading completed!")
            
            if torch.cuda.is_available():
                gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                device_str = "cuda" if str(gpu_device) in ("auto", "cuda") else f"cuda:{gpu_device}"
                try:
                    pipeline = pipeline.to(device_str)
                    print(f"[FLUX LOADING] 🔄 Moved full-precision pipeline to {device_str}")
                except Exception as e:
                    print(f"[FLUX LOADING] ⚠️ Could not move pipeline to {device_str}: {e}")
        
        # Cache and return final result  
        self._pipeline_cache[cache_key] = pipeline
        print(f"[FLUX DEBUG] About to return pipeline: {pipeline is not None} (type: {type(pipeline)})")
        
        # Return the final result directly
        return pipeline
    
    def _get_pipeline(self, model_id: str, quantization: str = "none", system_constraints: dict = None) -> Any:
        """Load diffusers pipeline with caching and optional quantization using shared backend"""
        cache_key = f"{model_id}_{quantization}"
        # Fail-fast validation of max_memory dict
        if system_constraints:
            system_constraints['max_memory'] = self._normalize_memory_dict(system_constraints.get('max_memory', {}), gpu_device=system_constraints.get('gpu_device', 0))

        # Intelligent cache management: check if we need to clear old models
        self._manage_pipeline_cache(cache_key, quantization, system_constraints)
        
        if cache_key in self._pipeline_cache:
            print(f"[FLUX LOADING] ✅ Using cached pipeline for {cache_key}")
            return self._pipeline_cache[cache_key]
        
        # Memory safety check before loading
        memory_check_result = self._check_memory_safety(quantization, system_constraints)
        if memory_check_result == "ASYNC_MEMORY_CHECK_NEEDED":
            # This signals that we need to do an async memory check
            # The calling code should handle this
            return "ASYNC_MEMORY_CHECK_NEEDED"
        
        # Use pre-loaded modules from shared backend
        if not self._shared_backend['available']:
            raise ImportError(f"Shared backend not available: {self._shared_backend.get('error', 'Unknown error')}")
        
        FluxPipeline = self._shared_backend['FluxPipeline']
        torch = self._shared_backend['torch']
        
        # Setup quantization config if needed using shared backend
        quantization_config = None
        bnb_config = None
        
        if quantization in ["4-bit", "8-bit"]:
            try:
                BitsAndBytesConfig = self._shared_backend['BitsAndBytesConfig']
                print(f"[FLUX DEBUG] BitsAndBytesConfig available, setting up {quantization} quantization...")
                
                if quantization == "4-bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    print(f"[FLUX DEBUG] ✅ Created 4-bit BitsAndBytesConfig")
                elif quantization == "8-bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        load_in_8bit_fp32_cpu_offload=True,  # Enable CPU offload for insufficient GPU memory
                        llm_int8_enable_fp32_cpu_offload=True,  # Additional CPU offload support
                        llm_int8_skip_modules=None,  # Don't skip any modules by default
                        llm_int8_threshold=6.0  # Standard threshold for outlier detection
                    )
                    print(f"[FLUX DEBUG] ✅ Created 8-bit BitsAndBytesConfig with enhanced CPU offload")
                
                # Try different approaches for quantization config
                quantization_config = None
                
                # Method 1: Try direct BitsAndBytesConfig (modern diffusers)
                try:
                    print(f"[FLUX DEBUG] 🔄 Trying direct BitsAndBytesConfig approach...")
                    quantization_config = bnb_config
                    print(f"[FLUX DEBUG] ✅ Using direct BitsAndBytesConfig for {quantization} quantization")
                except Exception as e1:
                    print(f"[FLUX DEBUG] ❌ Direct BitsAndBytesConfig failed: {e1}")
                    
                # Method 2: Try PipelineQuantizationConfig if direct approach failed
                if quantization_config is None:
                    try:
                        from diffusers import PipelineQuantizationConfig
                        quantization_config = PipelineQuantizationConfig(
                            quant_backend="bitsandbytes",
                            quant_kwargs=bnb_config.__dict__,
                            components_to_quantize=["transformer"]
                        )
                        print(f"[FLUX DEBUG] ✅ Using PipelineQuantizationConfig for {quantization} quantization")
                    except Exception as e2:
                        print(f"[FLUX DEBUG] ❌ PipelineQuantizationConfig failed: {e2}")
                        
                # Method 3: Disable quantization if all methods fail
                if quantization_config is None:
                    print(f"[FLUX DEBUG] ⚠️ All quantization methods failed, disabling quantization")
                    bnb_config = None
                    
            except Exception as e:
                print(f"[FLUX ERROR] ❌ Quantization setup failed: {e}")
                print(f"[FLUX ERROR] ❌ Falling back to full precision - this will use ~25GB+ memory!")
                print(f"[FLUX ERROR] 💡 To fix: pip install bitsandbytes>=0.41.0")
                quantization_config = None
                bnb_config = None
        
        # Load pipeline - try different quantization approaches
        pipeline = None
        
        if quantization_config is not None:
            # Method 1: Try with component-level quantization
            try:
                print(f"[FLUX LOADING] 🔄 Loading {model_id} with component-level quantization ({quantization})...")
                print(f"[FLUX LOADING] 💾 This should use much less memory than full precision")
                
                # Load transformer with quantization
                from diffusers import FluxTransformer2DModel
                
                # Set up loading parameters based on quantization type
                # Unified approach for all quantization types
                loading_kwargs = {
                    "subfolder": "transformer",
                    "quantization_config": quantization_config,
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True
                }
                
                # Optimize device mapping for quantization (4-bit and 8-bit)
                if quantization in ["4-bit", "8-bit"]:
                    # Use auto device mapping with memory constraints for quantization
                    gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                    
                    # Handle 'auto' device selection - convert to device number
                    if gpu_device == 'auto':
                        gpu_device = 0
                    
                    # Keep device_map as string - "auto" works best with quantization
                    loading_kwargs["device_map"] = "auto"
                    
                    # Add memory constraints to enable automatic CPU offload when needed
                    if system_constraints:
                        max_memory = system_constraints.get('max_memory', {})
                        if max_memory:
                            # Convert 'auto' device keys to numbers for transformer loading
                            fixed_memory = {}
                            for key, value in max_memory.items():
                                if key == 'auto':
                                    fixed_memory[gpu_device] = value
                                elif isinstance(key, int):
                                    fixed_memory[key] = value
                                elif isinstance(key, str) and key.isdigit():
                                    fixed_memory[int(key)] = value
                                elif isinstance(key, str) and key.startswith("cuda:"):
                                    fixed_memory[key] = value
                                else:
                                    fixed_memory[key] = value
                            loading_kwargs["max_memory"] = fixed_memory
                            # Remove device_map when explicit memory dict is provided to avoid key casting
                            loading_kwargs.pop("device_map", None)
                            print(f"[FLUX LOADING] {quantization} with memory constraints: {fixed_memory}")
                    else:
                        # Default conservative memory limits to ensure CPU offload works
                        if quantization == "4-bit":
                            # More conservative for 4-bit to ensure all fits
                            loading_kwargs["max_memory"] = {f"cuda:{gpu_device}": "8GB", "cpu": "32GB"}
                        else:  # 8-bit
                            loading_kwargs["max_memory"] = {f"cuda:{gpu_device}": "12GB", "cpu": "32GB"}
                        print(f"[FLUX LOADING] {quantization} with default memory limits for device {gpu_device}")
                
                print(f"[FLUX LOADING] Using component-level quantization for {quantization}")
                
                transformer = FluxTransformer2DModel.from_pretrained(model_id, **loading_kwargs)
                
                # Load pipeline with quantized transformer
                # For 8-bit quantization, ensure VAE is loaded on GPU from the start
                pipeline_kwargs = {
                    "transformer": transformer,
                    "torch_dtype": torch.bfloat16,
                    "use_safetensors": True,
                }
                
                if quantization in ["4-bit", "8-bit"] and torch.cuda.is_available():
                    # For quantized models, let accelerate handle placement but with memory constraints
                    gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                    max_memory = system_constraints.get('max_memory', {}) if system_constraints else {}
                    
                    # Modify memory constraints to encourage VAE on GPU
                    if max_memory:
                        # Reduce CPU memory to force VAE onto GPU and fix 'auto' device keys
                        modified_memory = {}
                        for key, value in max_memory.items():
                            if key == 'auto':
                                # Convert 'auto' to device number 0
                                modified_memory[gpu_device] = value
                            elif key == 'cpu':
                                # Reduce CPU memory to force VAE to GPU
                                modified_memory['cpu'] = "8GB"  # Smaller CPU allocation
                            else:
                                modified_memory[key] = value
                        pipeline_kwargs.update({
                            "device_map": "balanced",  # FluxPipeline only supports "balanced"
                            "max_memory": modified_memory,
                            "low_cpu_mem_usage": True
                        })
                        print(f"[FLUX LOADING] {quantization}: Using modified memory constraints with balanced device map")
                    else:
                        # Use default constraints that favor VAE on GPU, adjusted per quantization
                        if quantization == "4-bit":
                            memory_config = {gpu_device: "6GB", "cpu": "8GB"}  # More conservative for 4-bit
                        else:  # 8-bit
                            memory_config = {gpu_device: "10GB", "cpu": "8GB"}  # Standard for 8-bit
                        
                        pipeline_kwargs.update({
                            "device_map": "balanced",  # FluxPipeline only supports "balanced"
                            "max_memory": memory_config,  # Force VAE to GPU
                            "low_cpu_mem_usage": True
                        })
                        print(f"[FLUX LOADING] {quantization}: Using conservative memory limits with balanced device map")
                
                pipeline = FluxPipeline.from_pretrained(model_id, **pipeline_kwargs)
                
                # Smart device placement with quantization support
                if torch.cuda.is_available():
                    gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                    device_str = "cuda" if str(gpu_device) in ("auto", "cuda") else f"cuda:{gpu_device}"
                    
                    if quantization in ["4-bit", "8-bit"]:
                        # For quantized models, check VAE placement and move if needed
                        print(f"[FLUX LOADING] 🔧 Checking VAE placement for {quantization} quantization...")
                        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                            vae_device = next(pipeline.vae.parameters()).device
                            print(f"[FLUX LOADING] VAE is currently on: {vae_device}")
                            
                            if 'cpu' in str(vae_device):
                                # VAE is on CPU - need to move it to GPU for decode compatibility
                                print(f"[FLUX LOADING] ⚠️ VAE on CPU will cause decode failures with 8-bit")
                                print(f"[FLUX LOADING] 🔧 Moving VAE to {device_str} (required for 8-bit decode)")
                                try:
                                    # This is the only safe way with 8-bit quantization
                                    pipeline.vae = pipeline.vae.to(device_str)
                                    print(f"[FLUX LOADING] ✅ VAE successfully moved to {device_str}")
                                except Exception as e:
                                    print(f"[FLUX LOADING] ❌ Could not move VAE: {e}")
                                    print(f"[FLUX LOADING] ❌ 8-bit quantization will likely fail during decode")
                            else:
                                print(f"[FLUX LOADING] ✅ VAE already on GPU - ready for decode")
                        else:
                            print(f"[FLUX LOADING] ⚠️ VAE not found - may cause decode issues")
                    else:
                        # For non-quantized models, try normal GPU placement
                        try:
                            print(f"[FLUX LOADING] 🎯 Attempting optimal GPU placement...")
                            pipeline = pipeline.to(device_str)
                            print(f"[FLUX LOADING] ✅ All components → {device_str}")
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"[FLUX LOADING] ⚠️ GPU memory insufficient for full pipeline")
                                
                                # Smart fallback - offload text encoders to CPU
                                print(f"[FLUX LOADING] 🔄 Applying automatic memory optimization...")
                                
                                try:
                                    # Move text encoders to CPU (usually saves 6-8GB)
                                    text_encoders_moved = []
                                    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                                        pipeline.text_encoder = pipeline.text_encoder.to("cpu")
                                        text_encoders_moved.append("CLIP")
                                    
                                    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
                                        pipeline.text_encoder_2 = pipeline.text_encoder_2.to("cpu")
                                        text_encoders_moved.append("T5")
                                    
                                    # Keep VAE on GPU for speed (small memory footprint)
                                    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                                        pipeline.vae = pipeline.vae.to(device_str)
                                        print(f"[FLUX LOADING] 🖥️ VAE → {device_str}")
                                        print(f"[FLUX LOADING] ⚡ Transformer → auto-distributed by accelerate")
                                        print(f"[FLUX LOADING] ✅ Memory optimization successful")
                                    else:
                                        print(f"[FLUX LOADING] ⚠️ No text encoders found to offload")
                                        
                                except Exception as fallback_error:
                                    print(f"[FLUX LOADING] ❌ Fallback strategy failed: {fallback_error}")
                                    print(f"[FLUX LOADING] 🏁 Using accelerate's automatic placement")
                            else:
                                # Non-memory related error, re-raise
                                raise e
                            
                elif quantization == "8-bit":
                    # For 8-bit quantization, let accelerate handle device placement automatically
                    # DO NOT manually move components - this causes device mismatch errors
                    print(f"[FLUX LOADING] 🎯 8-bit quantization loaded successfully")
                    print(f"[FLUX LOADING] ✅ Components automatically distributed by accelerate")
                    
                    # Debug: Report memory usage without interfering with placement
                    if torch.cuda.is_available():
                        gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                        
                        # Handle 'auto' device selection
                        if gpu_device == 'auto':
                            gpu_device = 0
                            
                        gpu_memory_gb = torch.cuda.get_device_properties(gpu_device).total_memory / 1024**3
                        gpu_allocated_gb = torch.cuda.memory_allocated(gpu_device) / 1024**3
                        gpu_reserved_gb = torch.cuda.memory_reserved(gpu_device) / 1024**3
                        print(f"[FLUX MEMORY] GPU {gpu_device} Total: {gpu_memory_gb:.1f}GB")
                        print(f"[FLUX MEMORY] GPU {gpu_device} Allocated: {gpu_allocated_gb:.1f}GB")
                        print(f"[FLUX MEMORY] GPU {gpu_device} Reserved: {gpu_reserved_gb:.1f}GB")
                        print(f"[FLUX MEMORY] GPU {gpu_device} Utilization: {gpu_reserved_gb/gpu_memory_gb*100:.1f}%")
                    
                    # Report component locations WITHOUT moving them
                    def get_component_device(component, name):
                        try:
                            if component is None:
                                return "Not present"
                            if hasattr(component, 'device'):
                                return str(component.device)
                            elif hasattr(component, 'parameters'):
                                param = next(component.parameters(), None)
                                return str(param.device) if param is not None else "No parameters"
                            else:
                                return "Unknown"
                        except Exception:
                            return "Error checking"
                    
                    print(f"[FLUX MEMORY] 📍 Component locations:")
                    if hasattr(pipeline, 'transformer'):
                        print(f"[FLUX MEMORY]   Transformer: {get_component_device(pipeline.transformer, 'transformer')}")
                    if hasattr(pipeline, 'vae'):
                        print(f"[FLUX MEMORY]   VAE: {get_component_device(pipeline.vae, 'vae')}")
                    if hasattr(pipeline, 'text_encoder'):
                        print(f"[FLUX MEMORY]   CLIP: {get_component_device(pipeline.text_encoder, 'text_encoder')}")
                    if hasattr(pipeline, 'text_encoder_2'):
                        print(f"[FLUX MEMORY]   T5: {get_component_device(pipeline.text_encoder_2, 'text_encoder_2')}")
                    
                    print(f"[FLUX MEMORY] ⚠️ For 8-bit quantization, letting accelerate handle device placement")
                    
                    # For 8-bit quantization, check that VAE will be accessible for decode
                    # Don't manually move quantized components - this can break quantization
                    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                        vae_location = get_component_device(pipeline.vae, 'vae')
                        print(f"[FLUX MEMORY] VAE location: {vae_location}")
                        
                        # Verify VAE is accessible for decode operations
                        if 'cpu' in vae_location:
                            print(f"[FLUX MEMORY] ⚠️ VAE on CPU - accelerate should handle decode automatically")
                            print(f"[FLUX MEMORY] ⚠️ If decode fails, this indicates a quantization device mapping issue")
                        else:
                            print(f"[FLUX MEMORY] ✅ VAE on GPU - should decode successfully")
                
                print(f"[FLUX LOADING] ✅ Successfully loaded with component-level quantization")
                
            except Exception as e1:
                print(f"[FLUX LOADING] ❌ Component-level quantization failed: {e1}")
                
                # Method 2: Pipeline-level quantization not supported for FluxPipeline
                print(f"[FLUX LOADING] 🔄 Pipeline-level quantization not supported for FLUX")
                print(f"[FLUX LOADING] ❌ Skipping pipeline-level quantization (not available for FLUX)")
                pipeline = None
        
        if pipeline is None:
            # Fallback to full precision
            print(f"[FLUX LOADING] ⚠️ Loading {model_id} without quantization (FULL PRECISION)")
            print(f"[FLUX LOADING] ⚠️ This will use ~25GB+ memory and may cause OOM!")
            pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            
            # Move to appropriate device for non-quantized models
            if torch.cuda.is_available():
                # Use GPU device from system constraints if available
                gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                device_str = "cuda" if str(gpu_device) in ("auto", "cuda") else f"cuda:{gpu_device}"
                
                try:
                    pipeline = pipeline.to(device_str)
                    print(f"[FLUX LOADING] 🔄 Moved full-precision pipeline to {device_str}")
                except Exception as e:
                    print(f"[FLUX LOADING] ⚠️ Could not move pipeline to {device_str}: {e}")
        
        self._pipeline_cache[cache_key] = pipeline
        return pipeline
    
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
        
        # Clear any cached GPU memory before loading model
        print(f"[FLUX MEMORY] Clearing GPU cache before model load...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.cuda.ipc_collect() if hasattr(torch.cuda, 'ipc_collect') else None
        torch.cuda.synchronize()

        # Wait for allocator fragmentation to clear\n        def _wait_for_gpu_free(min_free_gb: float = 4.0, timeout: float = 30):\n            total = torch.cuda.get_device_properties(0).total_memory / 1024**3\n            import time\n            start = time.time()\n            while time.time() - start < timeout:\n                free_gb = (total - (torch.cuda.memory_allocated(0)+ torch.cuda.memory_reserved(0)) / 1024**3)\n                if free_gb >= min_free_gb:\n                    print(f'[FLUX MEMORY] {free_gb:.1f}GB free after cleanup')\n                    return\n                time.sleep(1)\n                torch.cuda.empty_cache()\n                torch.cuda.ipc_collect() if hasattr(torch.cuda,'ipc_collect') else None\n            print('[FLUX MEMORY] Timeout waiting for free VRAM; proceeding anyway')\n        _wait_for_gpu_free()
        import gc
        gc.collect()

        # Get pipeline – automatic fallback if bitsandbytes lacks kernels (e.g. sm90)
        try:
            pipeline = self._get_pipeline_with_progress(model_id, quantization, system_constraints)
        except RuntimeError as e:
            if "no kernel image" in str(e).lower():
                print("[FLUX FALLBACK] bitsandbytes kernel not available for this GPU – retrying full precision …")
                quantization = "none"
                pipeline = self._get_pipeline_with_progress(model_id, "none", system_constraints)
            else:
                raise
        
        # Handle async memory check if needed
        if pipeline == "ASYNC_MEMORY_CHECK_NEEDED":
            # Need to do async memory check - this requires generator conversion
            raise RuntimeError("ASYNC_MEMORY_CHECK_NEEDED")
        
        # Debug prompt parameter early
        print(f"[FLUX DEBUG] ===== PROMPT DEBUG =====")
        print(f"[FLUX DEBUG] Received prompt: '{prompt}'")
        print(f"[FLUX DEBUG] Pipeline object: {pipeline}")
        print(f"[FLUX DEBUG] Pipeline type: {type(pipeline)}")
        scheduler_name = kwargs.get('scheduler', 'DPM++ 2M Karras')
        cfg_rescale = kwargs.get('cfg_rescale', 0.0)
        denoise_eta = kwargs.get('denoise_eta', 0.0)
        # Apply scheduler if requested
        try:
            from diffusers import (
                DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler, DDIMScheduler, FlowMatchEulerDiscreteScheduler
            )
            if scheduler_name == "DPM++ 2M Karras":
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
            elif scheduler_name == "Euler A":
                pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
            elif scheduler_name == "Euler":
                pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            elif scheduler_name == "DDIM":
                pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            elif scheduler_name == "FlowMatchEuler":
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
            "guidance_rescale": cfg_rescale,
            "generator": torch.Generator().manual_seed(seed),
        }
        if denoise_eta > 0:
            generation_kwargs["eta"] = denoise_eta
        
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
            
            # Direct inference call - may briefly disconnect but callback shows progress
            try:
                backend_result = pipeline(**generation_kwargs)
                print(f"[FLUX INFERENCE] ✅ Pipeline call completed!")
            except RuntimeError as e:
                # Catch missing kernel errors from bitsandbytes after forward pass starts
                if "no kernel image" in str(e).lower() and quantization in ["4-bit", "8-bit"]:
                    # On Hopper GPUs the 8-bit kernels may be missing – retry with 4-bit to stay quantized
                    print("[FLUX FALLBACK] Kernel error – retrying in 4-bit …")
                    quantization_retry = "4-bit"
                    pipeline_fp = self._get_pipeline_with_progress(model_id, quantization_retry, system_constraints)
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
            return (generated_image, generation_info)
        else:
            print(f"[FLUX DEBUG] Unexpected backend result format: {type(backend_result)}")
            raise RuntimeError(f"Backend returned unexpected format: {type(backend_result)}") 
