"""
FLUX CUDA backend coordinator.

Coordinates between memory management, pipeline loading, and generation components.
"""

import os
import sys
import subprocess
import warnings
from typing import Any

# Import backend components with fallback for standalone loading
try:
    from .flux_memory_manager import FluxMemoryManager
    from .flux_pipeline_loader import FluxPipelineLoader
    from .flux_generation_engine import FluxGenerationEngine
except ImportError:
    # Fallback for standalone loading
    import importlib.util
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load FluxMemoryManager
    spec = importlib.util.spec_from_file_location("flux_memory_manager", os.path.join(current_dir, "flux_memory_manager.py"))
    memory_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory_module)
    FluxMemoryManager = memory_module.FluxMemoryManager
    
    # Load FluxPipelineLoader
    spec = importlib.util.spec_from_file_location("flux_pipeline_loader", os.path.join(current_dir, "flux_pipeline_loader.py"))
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    FluxPipelineLoader = pipeline_module.FluxPipelineLoader
    
    # Load FluxGenerationEngine
    spec = importlib.util.spec_from_file_location("flux_generation_engine", os.path.join(current_dir, "flux_generation_engine.py"))
    engine_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine_module)
    FluxGenerationEngine = engine_module.FluxGenerationEngine


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


class FluxBackend:
    """Abstract base class for FLUX generation backends"""
    
    def is_available(self) -> bool:
        """Check if this backend is available on current system"""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Get backend name for display"""
        raise NotImplementedError
    
    def generate_image(self, **kwargs):
        """Generate image and return (pil_image, generation_info) - can be generator"""
        raise NotImplementedError
    
    def validate_model_id(self, model_id: str) -> bool:
        """Check if model_id is supported by this backend"""
        raise NotImplementedError


class DiffusersFluxBackend(FluxBackend):
    """Diffusers-based backend for CUDA/CPU using shared backend"""
    
    def __init__(self):
        """Initialize the FluxInference node with optimized GPU memory management"""
        super().__init__()
        
        # Configure PyTorch CUDA allocator via environment variables (safe before torch import)
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
                            'garbage_collection_threshold:0.8,max_split_size_mb:128,expandable_segments:True')
        
        print(f"[FLUX INIT] 🔧 Configured CUDA allocator for improved memory management")
        print(f"[FLUX INIT] 🔧 CUDA allocator config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
        print(f"[FLUX INIT] 💡 Advanced users: Set PYTORCH_CUDA_ALLOC_CONF env var to customize memory management")
        
        # Ensure bitsandbytes kernels are present (installs wheel if needed)
        _ensure_bitsandbytes()
        
        # Load configuration
        try:
            from .flux_config import FluxConfig
            self._config = FluxConfig()
            print(f"[FLUX CONFIG] Configuration loaded successfully")
        except Exception as e:
            print(f"[FLUX CONFIG] Warning: Could not load config, using defaults: {e}")
            self._config = None
        
        # Initialize shared backend for heavy library imports
        self._shared_backend = self._get_shared_backend()
        
        # Initialize component managers with config
        self._memory_manager = FluxMemoryManager(self._shared_backend, self._config)
        self._pipeline_loader = FluxPipelineLoader(self._shared_backend, self._memory_manager, self._config)
        self._generation_engine = FluxGenerationEngine(self._shared_backend, self._memory_manager, self._pipeline_loader, self._config)
    
    def _get_shared_backend(self):
        """Get the shared backend from the advanced library"""
        try:
            # Try relative import first (when loaded as package)
            print("[FLUX DEBUG] Trying relative import...")
            try:
                from .. import get_shared_backend as _get_shared_backend
            except ImportError:
                # Fallback for standalone loading
                import importlib.util
                import os
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                spec = importlib.util.spec_from_file_location("library_loader", os.path.join(parent_dir, "library_loader.py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _get_shared_backend = module.get_shared_backend
            result = _get_shared_backend()
            print("[FLUX DEBUG] ✅ Relative import successful")
            return result
        except ImportError as e1:
            print(f"[FLUX DEBUG] Relative import failed: {e1}")
            try:
                # Method 1: Direct import from advanced library
                print("[FLUX DEBUG] Trying direct advanced library import...")
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
        """Check if model_id is supported by this backend"""
        # Accept any FLUX.1 model variants (dev, schnell, or community versions like Krea)
        return "FLUX.1" in model_id
    
    def test_quantization_setup(self, quantization: str) -> bool:
        """Test quantization setup without loading models"""
        return self._pipeline_loader.test_quantization_setup(quantization)
    
    def get_pipeline_with_progress(self, model_id: str, quantization: str = "none", system_constraints: dict = None):
        """Load pipeline with progress tracking"""
        return self._pipeline_loader.get_pipeline_with_progress(model_id, quantization, system_constraints)
    
    def get_pipeline(self, model_id: str, quantization: str = "none", system_constraints: dict = None):
        """Load pipeline with caching and optional quantization"""
        return self._pipeline_loader.get_pipeline(model_id, quantization, system_constraints)
    
    def generate_image(self, **kwargs):
        """Generate image using the generation engine (with async loading support)"""
        generation_result = self._generation_engine.generate_image(**kwargs)
        
        # Handle generator result from async loading
        if str(type(generation_result).__name__) == 'generator':
            # It's a generator - yield progress and return final result
            for step in generation_result:
                if isinstance(step, tuple):
                    # Final result (image, info)
                    yield step
                    return
                else:
                    # Progress step - yield to maintain Griptape connection
                    yield f"[BACKEND] {step}"
        else:
            # Direct result
            yield generation_result