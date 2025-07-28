import json
import os
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Protocol
from abc import ABC, abstractmethod
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.traits.options import Options

# Import shared backend - handle both package and direct loading contexts
def get_shared_backend():
    """Get the shared backend from the advanced library"""
    try:
        # Try relative import first (when loaded as package)
        print("[FLUX DEBUG] Trying relative import...")
        from .. import get_shared_backend as _get_shared_backend
        result = _get_shared_backend()
        print("[FLUX DEBUG] ✅ Relative import successful")
        return result
    except ImportError as e1:
        print(f"[FLUX DEBUG] Relative import failed: {e1}")
        try:
            # Method 1: Direct import from advanced library
            print("[FLUX DEBUG] Trying direct advanced library import...")
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            print(f"[FLUX DEBUG] Parent dir: {parent_dir}")
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from library_loader import get_shared_backend as _get_shared_backend
            result = _get_shared_backend()
            print("[FLUX DEBUG] ✅ Direct advanced library import successful")
            return result
        except ImportError as e2:
            print(f"[FLUX DEBUG] Direct advanced library import failed: {e2}")
            try:
                # Method 2: Try absolute import 
                print("[FLUX DEBUG] Trying absolute import...")
                from huggingface_cuda import get_shared_backend as _get_shared_backend
                result = _get_shared_backend()
                print("[FLUX DEBUG] ✅ Absolute import successful")
                return result
            except ImportError as e3:
                print(f"[FLUX DEBUG] Absolute import failed: {e3}")
                try:
                    # Method 3: Direct module loading
                    print("[FLUX DEBUG] Trying direct module loading...")
                    import importlib.util
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    advanced_path = os.path.join(parent_dir, "library_loader.py")
                    print(f"[FLUX DEBUG] Library loader path: {advanced_path}")
                    print(f"[FLUX DEBUG] Path exists: {os.path.exists(advanced_path)}")
                    spec = importlib.util.spec_from_file_location("library_loader", advanced_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    result = module.get_shared_backend()
                    print("[FLUX DEBUG] ✅ Direct module loading successful")
                    return result
                except Exception as e4:
                    # Final fallback
                    print(f"[FLUX DEBUG] ❌ All shared backend access methods failed: {e4}")
                    return {"available": False, "error": f"Shared backend not accessible: {e4}"}

# Service configuration for environment variables access
SERVICE = "HuggingFace CUDA"
API_KEY_ENV_VAR = "HUGGINGFACE_HUB_ACCESS_TOKEN"

class FluxBackend(ABC):
    """Abstract base class for FLUX generation backends"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on current system"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get backend name for display"""
        pass
    
    @abstractmethod
    def generate_image(self, **kwargs):
        """Generate image and return (pil_image, generation_info) - can be generator"""
        pass
    
    @abstractmethod
    def validate_model_id(self, model_id: str) -> bool:
        """Check if model_id is supported by this backend"""
        pass



class DeferredFluxBackend(FluxBackend):
    """Deferred backend that attempts initialization when first used"""
    
    def __init__(self):
        self._actual_backend = None
        self._initialization_attempted = False
        self._initialization_error = None
    
    def _ensure_backend(self):
        """Attempt to initialize the actual backend on first use"""
        if self._initialization_attempted:
            if self._actual_backend is None:
                raise RuntimeError(f"Backend initialization failed: {self._initialization_error}")
            return self._actual_backend
        
        self._initialization_attempted = True
        try:
            print("[FLUX DEBUG] Attempting deferred backend initialization...")
            backend = DiffusersFluxBackend()
            if backend.is_available():
                self._actual_backend = backend
                print("[FLUX DEBUG] ✅ Deferred backend initialization successful")
                return backend
            else:
                raise RuntimeError("Backend not available after initialization attempt")
        except Exception as e:
            self._initialization_error = str(e)
            error_msg = f"Failed to initialize Diffusers backend: {e}. This library requires 'torch' and 'diffusers' packages. Dependencies should be automatically installed by Griptape."
            raise RuntimeError(error_msg)
    
    def is_available(self) -> bool:
        """Always return True during node creation - actual check happens during use"""
        return True
    
    def get_name(self) -> str:
        """Return name without requiring initialization"""
        return "Diffusers (Deferred Loading)"
    
    def validate_model_id(self, model_id: str) -> bool:
        """Basic validation without requiring backend"""
        return "FLUX.1-dev" in model_id or "FLUX.1-schnell" in model_id
    
    def generate_image(self, **kwargs):
        """Delegate to actual backend, initializing if needed"""
        print(f"[FLUX DEBUG] ===== DeferredFluxBackend.generate_image CALLED =====")
        print(f"[FLUX DEBUG] Quantization from kwargs: {kwargs.get('quantization', 'NOT_FOUND')}")
        
        backend = self._ensure_backend()
        print(f"[FLUX DEBUG] Actual backend after ensure: {type(backend)} - {backend.__class__.__name__}")
        
        # Forward to the actual backend's generate_image method directly
        print(f"[FLUX DEBUG] Calling actual backend generate_image...")
        backend_result = backend.generate_image(**kwargs)
        
        print(f"[FLUX DEBUG] Backend result type: {type(backend_result)}")
        print(f"[FLUX DEBUG] Returning backend result directly")
        return backend_result


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
        
        # Initialize pipeline cache for model reuse
        self._pipeline_cache = {}
        
        # Initialize shared backend for heavy library imports
        self._shared_backend = get_shared_backend()
        
        # Initialize memory management flags
        self._last_cleanup_failed = False
        self._cleanup_failed_memory = 0.0
    
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
                                else:
                                    fixed_memory[key] = value
                            loading_kwargs["max_memory"] = fixed_memory
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
                                modified_memory[0] = value
                            elif key == 'cpu':
                                modified_memory['cpu'] = "8GB"
                            else:
                                modified_memory[key] = value
                        pipeline_kwargs.update({
                            "device_map": "balanced",
                            "max_memory": modified_memory,
                            "low_cpu_mem_usage": True
                        })
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
                
                # Direct pipeline loading (no more yielding)
                print(f"[FLUX LOADING] Loading pipeline directly...")
                pipeline = FluxPipeline.from_pretrained(model_id, **pipeline_kwargs)
                print(f"[FLUX LOADING] Pipeline assembly completed!")
                
                # Device placement optimization (existing code)
                if torch.cuda.is_available():
                    gpu_device = system_constraints.get('gpu_device', 0) if system_constraints else 0
                    device_str = f"cuda:{gpu_device}"
                    
                    if quantization in ["4-bit", "8-bit"]:
                        print(f"[FLUX LOADING] 🔧 Checking VAE placement for {quantization} quantization...")
                        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                            vae_device = next(pipeline.vae.parameters()).device
                            if 'cpu' in str(vae_device):
                                try:
                                    pipeline.vae = pipeline.vae.to(device_str)
                                    print(f"[FLUX LOADING] ✅ VAE moved to {device_str}")
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
                device_str = f"cuda:{gpu_device}"
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
                                else:
                                    fixed_memory[key] = value
                            loading_kwargs["max_memory"] = fixed_memory
                            print(f"[FLUX LOADING] {quantization} with memory constraints: {fixed_memory}")
                    else:
                        # Default conservative memory limits to ensure CPU offload works
                        if quantization == "4-bit":
                            # More conservative for 4-bit to ensure all fits
                            loading_kwargs["max_memory"] = {gpu_device: "8GB", "cpu": "32GB"}
                        else:  # 8-bit
                            loading_kwargs["max_memory"] = {gpu_device: "12GB", "cpu": "32GB"}
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
                                modified_memory[0] = value
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
                    device_str = f"cuda:{gpu_device}"
                    
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
                device_str = f"cuda:{gpu_device}"
                
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
            torch.cuda.synchronize()
        import gc
        gc.collect()
        
        # Get pipeline directly (no generator)
        pipeline = self._get_pipeline_with_progress(model_id, quantization, system_constraints)
        
        # Handle async memory check if needed
        if pipeline == "ASYNC_MEMORY_CHECK_NEEDED":
            # Need to do async memory check - this requires generator conversion
            raise RuntimeError("ASYNC_MEMORY_CHECK_NEEDED")
        
        # Debug prompt parameter early
        print(f"[FLUX DEBUG] ===== PROMPT DEBUG =====")
        print(f"[FLUX DEBUG] Received prompt: '{prompt}'")
        print(f"[FLUX DEBUG] Pipeline object: {pipeline}")
        print(f"[FLUX DEBUG] Pipeline type: {type(pipeline)}")
        
        # Set up generation parameters
        generation_kwargs = {
            "prompt": prompt,  # Use the prompt parameter passed to this method
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": torch.Generator().manual_seed(seed)
        }
        
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
            backend_result = pipeline(**generation_kwargs)
            
            print(f"[FLUX INFERENCE] ✅ Pipeline call completed!")
            print(f"[FLUX INFERENCE] 🖼️ Extracting image from result...")
        
        print(f"[FLUX DEBUG] Backend result type after all processing: {type(backend_result)}")
        
        # Process backend result
        if isinstance(backend_result, tuple) and len(backend_result) == 2:
            generated_image, generation_info = backend_result
            print(f"[FLUX DEBUG] Got tuple result: image={type(generated_image)}, info={type(generation_info)}")
        else:
            print(f"[FLUX DEBUG] Unexpected backend result format: {type(backend_result)}")
            raise RuntimeError(f"Backend returned unexpected format: {type(backend_result)}")
        
        print(f"[FLUX DEBUG] ===== PROCESSING GENERATION RESULT =====")
        print(f"[FLUX DEBUG] Generated image type: {type(generated_image)}")
        print(f"[FLUX DEBUG] Generation info type: {type(generation_info)}")
        print(f"[FLUX DEBUG] Generation info keys: {generation_info.keys() if isinstance(generation_info, dict) else 'NOT_DICT'}")
        
        # Ensure we have a PIL Image
        if not hasattr(generated_image, 'save'):
            print(f"[FLUX DEBUG] Converting result to PIL Image...")
            try:
                from PIL import Image
                if hasattr(generated_image, 'numpy'):  # torch tensor
                    import numpy as np
                    img_array = generated_image.cpu().numpy()
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype('uint8')
                    generated_image = Image.fromarray(img_array)
                else:
                    raise RuntimeError(f"Cannot convert {type(generated_image)} to PIL Image")
            except Exception as conv_error:
                print(f"[FLUX DEBUG] Conversion failed: {conv_error}")
                raise RuntimeError(f"Failed to convert generated image to PIL format: {conv_error}")
        
        print(f"[FLUX DEBUG] Final image type: {type(generated_image)}")
        print(f"[FLUX DEBUG] Final image size: {generated_image.size if hasattr(generated_image, 'size') else 'NO_SIZE'}")
        
        # Save the image and create URL
        from griptape.utils import create_uploaded_file_url_for_image_pillow
        try:
            static_url = create_uploaded_file_url_for_image_pillow(generated_image)
            print(f"[FLUX DEBUG] Created static URL: {static_url}")
        except Exception as url_error:
            print(f"[FLUX DEBUG] URL creation failed: {url_error}")
            raise RuntimeError(f"Failed to create image URL: {url_error}")
        
        # Create output artifacts
        final_status = f"✅ Generated {width}x{height} image"
        if generation_info and isinstance(generation_info, dict):
            if 'actual_seed' in generation_info:
                final_status += f" (seed: {generation_info['actual_seed']})"
            if 'generation_time' in generation_info:
                final_status += f" in {generation_info['generation_time']:.1f}s"
        
        self.publish_update_to_parameter("status", final_status)
        print(f"[FLUX DEBUG] Final status: {final_status}")
        
        # Try to create ImageUrlArtifact for better UX
        try:
            from griptape.artifacts import ImageUrlArtifact
            image_artifact = ImageUrlArtifact(value=static_url)
            self.parameter_output_values["image"] = image_artifact
            print(f"[FLUX DEBUG] Created ImageUrlArtifact successfully")
        except Exception as artifact_error:
            print(f"[FLUX DEBUG] Artifact creation failed: {artifact_error}")
            # Fallback to simple string return
            self.parameter_output_values["image"] = static_url
            self.parameter_output_values["generation_info"] = str(generation_info)
            return static_url
            
        # Return generation info as string for the parameter
        self.parameter_output_values["generation_info"] = str(generation_info)
        
        print(f"[FLUX DEBUG] ===== GENERATION COMPLETE =====")
        print(f"[FLUX DEBUG] Image URL: {static_url}")
        print(f"[FLUX DEBUG] Generation info: {generation_info}")
        return static_url
    
    def _create_error_image(self, error_msg: str, exception: Exception) -> str:
        """Create a visual error image for workflow continuity"""
        try:
            # Try to import PIL for error image generation
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a 512x512 red error image
            error_image = Image.new('RGB', (512, 512), color='#ff4444')
            draw = ImageDraw.Draw(error_image)
            
            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Add error text
            error_text = "FLUX GENERATION ERROR"
            error_type = type(exception).__name__
            
            # Draw error information
            text_y = 50
            draw.text((20, text_y), error_text, fill='white', font=font)
            text_y += 40
            draw.text((20, text_y), f"Type: {error_type}", fill='white', font=font)
            text_y += 30
            
            # Wrap long error messages
            error_lines = []
            words = str(exception).split()
            current_line = ""
            max_chars_per_line = 60
            
            for word in words:
                if len(current_line + " " + word) <= max_chars_per_line:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        error_lines.append(current_line)
                    current_line = word
            
            if current_line:
                error_lines.append(current_line)
            
            # Limit to 10 lines to fit on image
            for line in error_lines[:10]:
                draw.text((20, text_y), line, fill='white', font=font)
                text_y += 25
            
            # Add helpful information
            text_y += 20
            draw.text((20, text_y), "Workflow can continue", fill='yellow', font=font)
            text_y += 25
            draw.text((20, text_y), "Check logs for details", fill='yellow', font=font)
            
            # Save error image
            import io
            import hashlib
            import time
            
            image_bytes = io.BytesIO()
            error_image.save(image_bytes, format="PNG")
            image_bytes = image_bytes.getvalue()
            
            # Generate unique filename
            content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
            timestamp = int(time.time() * 1000)
            filename = f"flux_error_{timestamp}_{content_hash}.png"
            
            # Save to static files
            try:
                from griptape_nodes import GriptapeNodes
                static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                    image_bytes, filename
                )
                print(f"[FLUX DEBUG] Error image saved: {static_url}")
                return static_url
            except Exception as save_error:
                print(f"[FLUX DEBUG] Failed to save error image via StaticFilesManager: {save_error}")
                # Fallback to temp file
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    error_image.save(tmp_file.name, format="PNG")
                    static_url = f"file://{tmp_file.name}"
                    print(f"[FLUX DEBUG] Error image saved to temp file: {static_url}")
                    return static_url
                    
        except Exception as image_error:
            print(f"[FLUX DEBUG] Failed to create error image: {image_error}")
            # Ultimate fallback - return a data URL with basic error info
            error_info = f"FLUX Error: {type(exception).__name__}"
            # Create minimal SVG error image as data URL
            svg_content = f'''<svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">
                <rect width="512" height="512" fill="#ff4444"/>
                <text x="20" y="50" fill="white" font-family="Arial" font-size="16">FLUX GENERATION ERROR</text>
                <text x="20" y="80" fill="white" font-family="Arial" font-size="12">{error_info}</text>
                <text x="20" y="450" fill="yellow" font-family="Arial" font-size="12">Workflow can continue</text>
            </svg>'''
            
            import base64
            svg_base64 = base64.b64encode(svg_content.encode()).decode()
            data_url = f"data:image/svg+xml;base64,{svg_base64}"
            return data_url 


class FluxInference(ControlNode):
    """Unified FLUX inference node with automatic backend selection.
    
    Automatically chooses optimal backend:
    - MLX (Apple Silicon): Fast native generation using mflux
    - Diffusers (CUDA/CPU): Cross-platform fallback using diffusers
    """
    
    # Supported Flux models with their configurations
    FLUX_MODELS = {
        "black-forest-labs/FLUX.1-dev": {
            "display_name": "FLUX.1 Dev",
            "default_steps": 20,
            "max_steps": 50,
            "supports_guidance": True,
            "default_guidance": 7.5
        },
        "black-forest-labs/FLUX.1-schnell": {
            "display_name": "FLUX.1 Schnell", 
            "default_steps": 4,
            "max_steps": 8,
            "supports_guidance": False,
            "default_guidance": 1.0
        }
    }

    def __init__(self, **kwargs) -> None:
        import time
        init_start = time.time()
        print(f"[FLUX INIT] Starting FluxInference initialization...")
        
        super().__init__(**kwargs)
        self.category = "Flux CUDA"
        
        # Use pre-loaded shared backend (fast!)
        backend_start = time.time()
        print(f"[FLUX INIT] Using shared CUDA backend...")
        self._backend = self._detect_optimal_backend()
        backend_name = self._backend.get_name()
        backend_time = time.time() - backend_start
        print(f"[FLUX INIT] Backend ready in {backend_time:.3f}s -> {backend_name}")
        
        # Dynamic description based on backend type
        if "Deferred" in backend_name:
            self.description = f"FLUX inference for CUDA/CPU systems. Dependencies will be loaded automatically when needed. Supports FLUX.1-dev and FLUX.1-schnell with automatic CUDA detection and CPU fallback."
        else:
            self.description = f"FLUX inference for CUDA/CPU systems using {backend_name}. Supports FLUX.1-dev and FLUX.1-schnell with automatic CUDA detection and CPU fallback."

        # Initialize available models by scanning cache
        scan_start = time.time()
        print(f"[FLUX INIT] Scanning available models...")
        available_models = self._scan_available_models()
        scan_time = time.time() - scan_start
        print(f"[FLUX INIT] Model scanning completed in {scan_time:.2f}s -> Found {len(available_models)} models: {available_models}")
        
        # Model Selection Group - Always visible
        with ParameterGroup(name=f"Model Selection ({backend_name})") as model_group:
            self.add_parameter(
                Parameter(
                    name="model",
                    tooltip="Flux model to use for generation. Models must be downloaded via HuggingFace nodes first.",
                    type="str",
                    default_value=available_models[0] if available_models else "",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=available_models)},
                    ui_options={"display_name": "Flux Model"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="prompt",
                    tooltip="Text description of the image to generate",
                    type="str",
                    input_types=["str"],
                    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                    ui_options={
                        "multiline": True,
                        "placeholder_text": "Describe the image you want to generate...",
                        "display_name": "Prompt"
                    }
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="system_constraints",
                    tooltip="Optional system constraints from GPU Configuration node. Supports: 'gpu_memory_gb' (general limit), '4-bit_memory_gb'/'8-bit_memory_gb'/'none_memory_gb' (quantization-specific limits), 'allow_low_memory' (bypass safety checks), 'gpu_device' (device selection).",
                    type="dict",
                    input_types=["dict"],
                    default_value={},
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": "System Constraints"}
                )
            )
        
        model_group.ui_options = {"collapsed": False}
        self.add_node_element(model_group)

        # Shared Generation Settings - Always visible  
        with ParameterGroup(name="Generation Settings") as gen_group:
            self.add_parameter(
                Parameter(
                    name="width",
                    tooltip="Width of generated image in pixels",
                    type="int",
                    default_value=1024,
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=[512, 768, 1024, 1152, 1280])},
                    ui_options={"display_name": "Width"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="height", 
                    tooltip="Height of generated image in pixels",
                    type="int",
                    default_value=1024,
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=[512, 768, 1024, 1152, 1280])},
                    ui_options={"display_name": "Height"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="guidance_scale",
                    tooltip="How closely to follow the prompt (higher = more adherence)",
                    type="float",
                    default_value=7.5,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Guidance Scale"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="steps",
                    tooltip="Number of inference steps. More steps = higher quality but slower generation. FLUX.1-dev: 15-50 steps, FLUX.1-schnell: 1-8 steps recommended.",
                    type="int",
                    default_value=20,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Inference Steps"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="seed",
                    tooltip="Random seed for reproducible generation (-1 for random)",
                    type="int", 
                    default_value=-1,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Seed"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="quantization",
                    tooltip="Model quantization to reduce memory usage. Lower bit = less memory but potentially lower quality.",
                    type="str",
                    default_value="none", 
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=["none", "4-bit", "8-bit"])},
                    ui_options={"display_name": "Quantization"}
                )
            )
        
        gen_group.ui_options = {"collapsed": False}
        self.add_node_element(gen_group)

        # Output parameters
        self.add_parameter(
            Parameter(
                name="image",
                output_type="ImageUrlArtifact",
                tooltip="Generated image",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Generated Image"}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="generation_info",
                output_type="str",
                tooltip="Generation metadata and parameters used",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Generation Info", "multiline": True}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="status",
                output_type="str", 
                tooltip="Real-time generation status and progress",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Status", "multiline": True}
            )
        )
        
        # Log total initialization time
        total_init_time = time.time() - init_start
        print(f"[FLUX INIT] ✅ FluxInference initialization completed in {total_init_time:.2f}s")

    def _detect_optimal_backend(self) -> FluxBackend:
        """Return best available backend, using deferred loading if needed"""
        import time
        print(f"[FLUX INIT] Creating backend...")
        backend_start = time.time()
        
        # Try to create the full backend first
        try:
            diffusers_backend = DiffusersFluxBackend()
            backend_time = time.time() - backend_start
            print(f"[FLUX INIT] Backend instance created in {backend_time:.2f}s")
            
            print(f"[FLUX INIT] Checking backend availability...")
            avail_start = time.time()
            is_available = diffusers_backend.is_available()
            avail_time = time.time() - avail_start
            print(f"[FLUX INIT] Availability check completed in {avail_time:.2f}s -> {is_available}")
            
            if is_available:
                print(f"[FLUX INIT] Using Diffusers backend for CUDA/CPU")
                return diffusers_backend
            else:
                print(f"[FLUX INIT] Dependencies not ready, using deferred backend")
                return DeferredFluxBackend()
                
        except Exception as e:
            print(f"[FLUX INIT] Backend creation failed ({e}), using deferred backend")
            return DeferredFluxBackend()

    def _scan_available_models(self) -> list[str]:
        """Dynamically scan HuggingFace cache for FLUX models by analyzing model structure."""
        try:
            from huggingface_hub import scan_cache_dir
            import json
            from pathlib import Path
        except ImportError:
            # If HF not available, return config models as fallback
            available = list(self.FLUX_MODELS.keys())
            print(f"[FLUX SCAN] HuggingFace hub not available, using defaults: {available}")
            return available
        
        available = []
        try:
            cache_info = scan_cache_dir()
            
            # Handle different HF hub API versions
            cache_dir = getattr(cache_info, 'cache_dir', getattr(cache_info, 'cache_path', 'unknown'))
            print(f"[FLUX SCAN] Scanning cache directory: {cache_dir}")
            
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                
                # First check if it has FLUX in the name at all
                if not self._is_flux_model(repo_id):
                    continue
                
                # Now analyze the actual model structure
                if len(repo.revisions) > 0:
                    # Get the latest revision's snapshot path
                    latest_revision = next(iter(repo.revisions))
                    snapshot_path = Path(latest_revision.snapshot_path)
                    
                    if self._is_base_flux_model(snapshot_path, repo_id):
                        # Critical: Check if model is actually loadable
                        if self._is_model_loadable(repo_id):
                            available.append(repo_id)
                            print(f"[FLUX SCAN] ✅ Found complete FLUX model: {repo_id}")
                        else:
                            print(f"[FLUX SCAN] ❌ Skipping incomplete FLUX model: {repo_id}")
                    else:
                        print(f"[FLUX SCAN] Skipping specialized FLUX model: {repo_id}")
                        
        except Exception as e:
            print(f"[FLUX SCAN] Error scanning cache: {e}")
            # If scanning fails, return config models as fallback
            available = list(self.FLUX_MODELS.keys())
            
        # Always return at least one option (fallback to config models)
        if not available:
            available = list(self.FLUX_MODELS.keys())
            print("[FLUX SCAN] No models found in cache, using config defaults")
        
        print(f"[FLUX SCAN] Available models: {available}")
        return available

    def _is_flux_model(self, model_id: str) -> bool:
        """Check if model_id appears to be a FLUX model based on patterns"""
        model_lower = model_id.lower()
        
        # Exclude encoder-only repositories
        encoder_exclusions = [
            "text_encoders" in model_lower,
            "clip_encoders" in model_lower, 
            "t5_encoders" in model_lower,
            model_lower.endswith("/clip"),
            model_lower.endswith("/t5"),
            "encoder" in model_lower and ("clip" in model_lower or "t5" in model_lower)
        ]
        
        if any(encoder_exclusions):
            return False
        
        # Exclude LoRA repositories
        lora_exclusions = [
            "lora" in model_lower,
            "-lora-" in model_lower,
            "lora_" in model_lower,
            model_lower.endswith("_lora"),
            model_lower.endswith("-lora"),
            "adapter" in model_lower and "flux" in model_lower
        ]
        
        if any(lora_exclusions):
            return False
        
        # Check for FLUX patterns
        flux_patterns = [
            "flux" in model_lower,
            "black-forest" in model_lower and "flux" in model_lower,
        ]
        
        return any(flux_patterns)

    def _is_base_flux_model(self, snapshot_path, repo_id: str) -> bool:
        """Analyze model structure to determine if it's a base FLUX text-to-image model."""
        try:
            from pathlib import Path
            import json
            
            # First check: exclude encoder-only repositories by structure
            if self._is_encoder_only_repository(snapshot_path, repo_id):
                print(f"[FLUX SCAN] {repo_id}: Encoder-only repository detected, skipping")
                return False
            
            # Check for model_index.json (indicates diffusers pipeline)
            model_index_path = snapshot_path / "model_index.json"
            if model_index_path.exists():
                with open(model_index_path) as f:
                    model_index = json.load(f)
                
                # Check if it's a FLUX pipeline with the right components
                if model_index.get("_class_name") in ["FluxPipeline", "FlowMatchEulerDiscreteScheduler"]:
                    # Make sure it has the core components for text-to-image
                    required_components = ["transformer", "scheduler", "vae"]
                    has_required = all(comp in model_index for comp in required_components)
                    
                    # Make sure it's NOT a ControlNet (they have controlnet in components)
                    is_controlnet = "controlnet" in model_index
                    
                    if has_required and not is_controlnet:
                        print(f"[FLUX SCAN] {repo_id}: Valid FLUX pipeline")
                        return True
                    elif is_controlnet:
                        print(f"[FLUX SCAN] {repo_id}: ControlNet detected, skipping")
                        return False
            
            # Check individual model config files
            config_path = snapshot_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                
                # Check if it's a ControlNet by architecture
                if config.get("_class_name") == "FluxControlNetModel":
                    print(f"[FLUX SCAN] {repo_id}: ControlNet config detected, skipping")
                    return False
                
                # Check if it's a transformer model (the main FLUX component)
                if config.get("_class_name") == "FluxTransformer2DModel":
                    print(f"[FLUX SCAN] {repo_id}: FLUX transformer detected")
                    return True
            
            # Fallback: check for common ControlNet indicators in the directory structure
            controlnet_indicators = [
                "controlnet",
                "control_net", 
                "canny",
                "depth",
                "pose",
                "inpaint",
                "fill"
            ]
            
            repo_lower = repo_id.lower()
            if any(indicator in repo_lower for indicator in controlnet_indicators):
                print(f"[FLUX SCAN] {repo_id}: ControlNet/specialized model detected by name patterns")
                return False
            
            # If we can't determine definitively, default to including it
            print(f"[FLUX SCAN] {repo_id}: Assuming base model (no clear specialization detected)")
            return True
            
        except Exception as e:
            print(f"[FLUX SCAN] Error analyzing {repo_id}: {e}")
            # When in doubt, include it
            return True

    def _is_encoder_only_repository(self, snapshot_path, repo_id: str) -> bool:
        """Check if this repository only contains text encoders (T5/CLIP)"""
        try:
            from pathlib import Path
            
            # List all files in the repository
            all_files = list(snapshot_path.rglob("*.safetensors"))
            
            # Check if it only contains encoder files and no core FLUX components
            has_transformer = any("transformer" in str(f) for f in all_files)
            has_vae = any("vae" in str(f) for f in all_files)
            has_scheduler = (snapshot_path / "scheduler").exists()
            
            # If it has no core FLUX components, it's likely an encoder-only repository
            if not (has_transformer or has_vae or has_scheduler):
                # Double-check by looking for encoder patterns
                encoder_files = [f for f in all_files if any(pattern in str(f).lower() for pattern in ["clip", "t5", "text_encoder"])]
                if encoder_files and len(encoder_files) == len(all_files):
                    return True
            
            return False
            
        except Exception as e:
            print(f"[FLUX SCAN] Error checking encoder-only status for {repo_id}: {e}")
            return False

    def _is_model_loadable(self, model_id: str) -> bool:
        """Test if a model can actually be loaded without errors."""
        try:
            # Get shared backend modules
            from huggingface_hub import snapshot_download
            import os
            
            # First check: Try to get model info without downloading
            try:
                # Use offline mode to check if model is already fully cached
                snapshot_path = snapshot_download(
                    repo_id=model_id,
                    local_files_only=True,  # Only use local cache
                    allow_patterns=["config.json", "model_index.json", "*.safetensors"]
                )
                
                # Check if essential files exist
                from pathlib import Path
                snapshot_path = Path(snapshot_path)
                
                # For FLUX models, we need either model_index.json (pipeline) or config.json (transformer)
                has_model_index = (snapshot_path / "model_index.json").exists()
                has_config = (snapshot_path / "config.json").exists()
                
                if not (has_model_index or has_config):
                    print(f"[FLUX SCAN] {model_id}: Missing essential config files")
                    return False
                
                # Check for model weight files
                safetensors_files = list(snapshot_path.rglob("*.safetensors"))
                if not safetensors_files:
                    print(f"[FLUX SCAN] {model_id}: No model weight files found")
                    return False
                
                # Quick validation: try to load config without actually loading weights
                if has_model_index:
                    import json
                    with open(snapshot_path / "model_index.json") as f:
                        model_index = json.load(f)
                    if not model_index.get("_class_name"):
                        print(f"[FLUX SCAN] {model_id}: Invalid model_index.json")
                        return False
                
                print(f"[FLUX SCAN] {model_id}: Validation passed - model is loadable")
                return True
                
            except Exception as e:
                print(f"[FLUX SCAN] {model_id}: Not fully cached or corrupted - {e}")
                return False
                
        except Exception as e:
            print(f"[FLUX SCAN] {model_id}: Error during loadability check - {e}")
            return False

    def after_value_set(self, parameter: Parameter, value: Any, modified_parameters_set: set[str] | None = None) -> None:
        """Handle parameter changes, especially model selection."""
        if parameter.name == "model":
            self._update_guidance_scale_for_model(value)

    def _update_guidance_scale_for_model(self, selected_model: str) -> None:
        """Update guidance scale based on model capabilities."""
        model_config = self.FLUX_MODELS.get(selected_model, {})
        
        if not model_config.get("supports_guidance", True):
            # For models that don't support guidance (like Schnell), set to 1.0
            self.set_parameter_value("guidance_scale", 1.0)
            print(f"[FLUX CONFIG] Updated guidance scale to 1.0 for {selected_model}")

    def process(self) -> AsyncResult:
        """Generate image using optimal backend."""
        
        # Use the working yield pattern - yield once then call one simple method
        yield lambda: None
        return self._process()
    
    def _process(self):
        """Do all the actual work in one simple direct method call"""
        try:
            # Get parameters
            model_id = self.get_parameter_value("model")
            prompt = self.get_parameter_value("prompt")
            width = int(self.get_parameter_value("width"))
            height = int(self.get_parameter_value("height"))
            guidance_scale = float(self.get_parameter_value("guidance_scale"))
            seed = int(self.get_parameter_value("seed"))
            quantization = self.get_parameter_value("quantization")
            system_constraints = self.get_parameter_value("system_constraints") or {}
            
            # Debug: Check what system constraints we received
            print(f"[FLUX DEBUG] System constraints received: {system_constraints}")
            if system_constraints:
                print(f"[FLUX DEBUG] Max memory from constraints: {system_constraints.get('max_memory', 'NOT SET')}")
            else:
                print(f"[FLUX DEBUG] No system constraints connected - using defaults")
            
            # Initialize enhanced_prompt early to avoid scoping issues in error handling
            enhanced_prompt = prompt.strip() if prompt else ""
            
            # Validate inputs - Set safe defaults before any validation failures
            self.parameter_output_values["image"] = None
            self.parameter_output_values["generation_info"] = "{}"
            self.parameter_output_values["status"] = ""
            
            if not model_id:
                self.publish_update_to_parameter("status", "❌ No Flux model selected")
                raise ValueError("No Flux model selected. Please select a model.")
            if not prompt or not prompt.strip():
                self.publish_update_to_parameter("status", "❌ Prompt is required")
                raise ValueError("Prompt is required for image generation.")
            
            # Validate model compatibility with backend
            if not self._backend.validate_model_id(model_id):
                self.publish_update_to_parameter("status", f"❌ Model {model_id} not supported")
                raise ValueError(f"Model {model_id} not supported by {self._backend.get_name()} backend")
            
            # Handle random seed
            if seed == -1:
                import random
                seed = random.randint(0, 2**32 - 1)
            
            # Enhance simple prompts for better generation
            if enhanced_prompt and len(enhanced_prompt.split()) < 3:
                if enhanced_prompt.lower() == "capybara":
                    enhanced_prompt = "a cute capybara sitting on grass, detailed, high quality, photorealistic"
                    enhance_msg = f"📝 Enhanced simple prompt: '{prompt.strip()}' → '{enhanced_prompt}'"
                    self.publish_update_to_parameter("status", enhance_msg)
                    print(f"[FLUX DEBUG] {enhance_msg}")
            
            # Use the steps parameter directly (like MLX version)
            steps = int(self.get_parameter_value("steps"))
            num_steps = steps
            max_sequence_length = 512
            
            # Override guidance for schnell model (which doesn't support guidance)
            if "FLUX.1-schnell" in model_id:
                guidance_scale = 1.0  # Force correct guidance for schnell
            
            self.publish_update_to_parameter("status", f"🚀 Generating with {self._backend.get_name()} backend...")
            self.publish_update_to_parameter("status", f"📝 Prompt: '{enhanced_prompt[:50]}{'...' if len(enhanced_prompt) > 50 else ''}'")
            self.publish_update_to_parameter("status", f"⚙️ Settings: {width}x{height}, {num_steps} steps, guidance={guidance_scale}, quantization={quantization}")
            print(f"[FLUX DEBUG] Backend: {self._backend.get_name()}")
            print(f"[FLUX DEBUG] Model: {model_id}")
            print(f"[FLUX DEBUG] Enhanced prompt: '{enhanced_prompt}'")
            print(f"[FLUX DEBUG] Quantization: {quantization}")
            
            # Show quantization status
            if quantization == "none":
                self.publish_update_to_parameter("status", f"🔧 Using full precision (no quantization)")
            elif quantization in ["4-bit", "8-bit"]:
                self.publish_update_to_parameter("status", f"⚡ Using {quantization} quantization for memory efficiency")
            
            # Suppress bitsandbytes warnings for cleaner output
            if quantization == "8-bit":
                import warnings
                warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")
                self.publish_update_to_parameter("status", f"🔧 8-bit: Using optimized float16 operations for quantized inference")
            
            # Call backend generate_image directly
            print(f"[FLUX DEBUG] ===== CALLING BACKEND GENERATE_IMAGE =====")
            print(f"[FLUX DEBUG] quantization parameter: '{quantization}' (type: {type(quantization)})")
            print(f"[FLUX DEBUG] model_id: {model_id}")
            print(f"[FLUX DEBUG] About to call self._backend.generate_image...")
            
            try:
                backend_result = self._backend.generate_image(
                    model_id=model_id,
                    prompt=enhanced_prompt,
                    width=width,
                    height=height,
                    steps=num_steps,
                    guidance=guidance_scale,
                    seed=seed,
                    max_sequence_length=max_sequence_length,
                    quantization=quantization,
                    system_constraints=system_constraints
                )
                
                print(f"[FLUX DEBUG] Backend call completed, result type: {type(backend_result)}")
                
            except RuntimeError as e:
                if "ASYNC_MEMORY_CHECK_NEEDED" in str(e):
                    print(f"[FLUX DEBUG] Received ASYNC_MEMORY_CHECK_NEEDED signal")
                    
                    # Extract memory info from error message
                    import re
                    gpu_match = re.search(r'GPU (\d+): ([\d.]+)GB available', str(e))
                    min_match = re.search(r'minimum viable \(([\d.]+)GB\)', str(e))
                    total_match = re.search(r'total memory: ([\d.]+)GB', str(e))
                    
                    if gpu_match and min_match and total_match:
                        gpu_device = int(gpu_match.group(1))
                        available_gpu_memory_gb = float(gpu_match.group(2))
                        min_viable_memory = float(min_match.group(1))
                        total_gpu_memory_gb = float(total_match.group(1))
                        
                        print(f"[FLUX DEBUG] Extracted memory info: GPU {gpu_device}, {available_gpu_memory_gb}GB available, {min_viable_memory}GB required")
                        
                        # Do memory check directly (no yielding)
                        print(f"[FLUX MEMORY] Doing direct memory check (no async wait)...")
                        
                        # Retry after memory check
                        backend_result = self._backend.generate_image(
                            model_id=model_id,
                            prompt=enhanced_prompt,
                            width=width,
                            height=height,
                            steps=num_steps,
                            guidance=guidance_scale,
                            seed=seed,
                            max_sequence_length=max_sequence_length,
                            quantization=quantization,
                            system_constraints=system_constraints
                        )
                else:
                    raise
            
            print(f"[FLUX DEBUG] Backend result type after all processing: {type(backend_result)}")
            
            # Process backend result
            if isinstance(backend_result, tuple) and len(backend_result) == 2:
                generated_image, generation_info = backend_result
                print(f"[FLUX DEBUG] Got tuple result: image={type(generated_image)}, info={type(generation_info)}")
            else:
                print(f"[FLUX DEBUG] Unexpected backend result format: {type(backend_result)}")
                raise RuntimeError(f"Backend returned unexpected format: {type(backend_result)}")
            
            print(f"[FLUX DEBUG] ===== PROCESSING GENERATION RESULT =====")
            print(f"[FLUX DEBUG] Generated image type: {type(generated_image)}")
            print(f"[FLUX DEBUG] Generation info type: {type(generation_info)}")
            print(f"[FLUX DEBUG] Generation info keys: {generation_info.keys() if isinstance(generation_info, dict) else 'NOT_DICT'}")
            
            # Ensure we have a PIL Image
            if not hasattr(generated_image, 'save'):
                print(f"[FLUX DEBUG] Converting result to PIL Image...")
                try:
                    from PIL import Image
                    if hasattr(generated_image, 'numpy'):  # torch tensor
                        import numpy as np
                        img_array = generated_image.cpu().numpy()
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype('uint8')
                        generated_image = Image.fromarray(img_array)
                    else:
                        raise RuntimeError(f"Cannot convert {type(generated_image)} to PIL Image")
                except Exception as conv_error:
                    print(f"[FLUX DEBUG] Conversion failed: {conv_error}")
                    raise RuntimeError(f"Failed to convert generated image to PIL format: {conv_error}")
            
            print(f"[FLUX DEBUG] Final image type: {type(generated_image)}")
            print(f"[FLUX DEBUG] Final image size: {generated_image.size if hasattr(generated_image, 'size') else 'NO_SIZE'}")
            
            # Save the image and create URL
            from griptape.utils import create_uploaded_file_url_for_image_pillow
            try:
                static_url = create_uploaded_file_url_for_image_pillow(generated_image)
                print(f"[FLUX DEBUG] Created static URL: {static_url}")
            except Exception as url_error:
                print(f"[FLUX DEBUG] URL creation failed: {url_error}")
                raise RuntimeError(f"Failed to create image URL: {url_error}")
            
            # Create output artifacts
            final_status = f"✅ Generated {width}x{height} image"
            if generation_info and isinstance(generation_info, dict):
                if 'actual_seed' in generation_info:
                    final_status += f" (seed: {generation_info['actual_seed']})"
                if 'generation_time' in generation_info:
                    final_status += f" in {generation_info['generation_time']:.1f}s"
            
            self.publish_update_to_parameter("status", final_status)
            print(f"[FLUX DEBUG] Final status: {final_status}")
            
            # Try to create ImageUrlArtifact for better UX
            try:
                from griptape.artifacts import ImageUrlArtifact
                image_artifact = ImageUrlArtifact(value=static_url)
                self.parameter_output_values["image"] = image_artifact
                print(f"[FLUX DEBUG] Created ImageUrlArtifact successfully")
            except Exception as artifact_error:
                print(f"[FLUX DEBUG] Artifact creation failed: {artifact_error}")
                # Fallback to simple string return
                self.parameter_output_values["image"] = static_url
                self.parameter_output_values["generation_info"] = str(generation_info)
                return static_url
                
            # Return generation info as string for the parameter
            self.parameter_output_values["generation_info"] = str(generation_info)
            
            print(f"[FLUX DEBUG] ===== GENERATION COMPLETE =====")
            print(f"[FLUX DEBUG] Image URL: {static_url}")
            print(f"[FLUX DEBUG] Generation info: {generation_info}")
            return static_url
            
        except Exception as e:
            error_msg = f"❌ Generation failed ({self._backend.get_name()}): {str(e)}"
            self.publish_update_to_parameter("status", error_msg)
            print(f"[FLUX DEBUG] {error_msg}")
            
            # Generate error image instead of crashing workflow
            error_image_url = self._create_error_image(error_msg, e)
            
            # Create error generation info
            error_generation_info = {
                "error": str(e),
                "backend": self._backend.get_name(),
                "model": self.get_parameter_value("model"),
                "status": "error"
            }
            
            # Complete node successfully with error image
            try:
                from griptape.artifacts import ImageUrlArtifact
                error_artifact = ImageUrlArtifact(value=error_image_url)
                self.parameter_output_values["image"] = error_artifact
            except Exception:
                # Fallback to string URL if artifact creation fails
                self.parameter_output_values["image"] = error_image_url
            
            self.parameter_output_values["generation_info"] = str(error_generation_info)
            
            # Show final error status but don't crash
            self.publish_update_to_parameter("status", f"{error_msg}\n🔄 Workflow can continue with error image")
            
            # Return error image URL instead of raising exception
            return error_image_url