import logging
import time
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from huggingface_hub import snapshot_download

# Import validation functions from shared module
from .shared.model_validation import (
    determine_checkpoint,
    validate_model_paths,
    inspect_model_configuration,
    validate_clip_model,
    validate_t5_model,
    validate_flux_transformer,
    check_bitsandbytes_cuda_support,
    validate_pipeline_components
)

# All validation functions are now imported from shared.model_validation module above

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Verify this module is being imported
print("[CUDA LIBRARY] 📦 Library loader module imported successfully")

# Global shared backend storage
_shared_backend = None

def _check_bitsandbytes_cuda_support():
    """Ensure bitsandbytes is usable on the current GPU – auto-installs the right wheel if required."""
    try:
        import importlib, contextlib, subprocess, sys, os
        
        # First try to import bitsandbytes - this might fail with specific CUDA kernel issues
        with contextlib.suppress(ImportError):
            import bitsandbytes as bnb
            print("[CUDA LIBRARY] ✅ bitsandbytes imported successfully")
            return True
        
        print("[CUDA LIBRARY] 🔧 bitsandbytes not available or has CUDA issues - attempting auto-install...")
        
        # Try to determine CUDA version
        cuda_version = None
        try:
            import torch
            if torch.cuda.is_available():
                # Get CUDA version from PyTorch
                cuda_version = torch.version.cuda.replace(".", "")[:3]  # e.g., "12.1" -> "121"
                print(f"[CUDA LIBRARY] Detected CUDA version: {cuda_version}")
        except Exception:
            print("[CUDA LIBRARY] ⚠️ Could not detect CUDA version")
        
        # Auto-install appropriate bitsandbytes wheel based on CUDA version
        if cuda_version:
            bnb_wheel_url = f"https://github.com/TimDettmers/bitsandbytes/releases/download/0.41.1/bitsandbytes-0.41.1+cu{cuda_version}-cp311-cp311-linux_x86_64.whl"
            print(f"[CUDA LIBRARY] Installing bitsandbytes wheel: {bnb_wheel_url}")
            
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--force-reinstall", bnb_wheel_url
            ])
            
            # Verify installation worked
            try:
                import bitsandbytes as bnb
                print("[CUDA LIBRARY] ✅ bitsandbytes installed and imported successfully")
                return True
            except ImportError as e:
                print(f"[CUDA LIBRARY] ❌ bitsandbytes installation failed: {e}")
                return False
        else:
            print("[CUDA LIBRARY] ❌ Cannot auto-install bitsandbytes without CUDA version")
            return False
    
    except Exception as e:
        print(f"[CUDA LIBRARY] ❌ Error with bitsandbytes setup: {e}")
        return False

def initialize_cuda_backend():
    """Initialize the CUDA backend with proper device and library setup"""
    import torch
    from transformers import BitsAndBytesConfig
    import gc
    import os
    
    # Use environment variable or detect GPU
    preferred_device = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    
    print(f"[CUDA LIBRARY] 🚀 Initializing CUDA backend on GPU {preferred_device}")
    
    # Clean GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    
    device = f"cuda:{preferred_device}" if torch.cuda.is_available() else "cpu"
    
    # Memory configuration
    max_memory = {preferred_device: "31GB", "cpu": "24GB"} if torch.cuda.is_available() else {"cpu": "24GB"}
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    backend_config = {
        "device": device,
        "device_map": "auto",
        "max_memory": max_memory,
        "quantization_config": quantization_config,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "use_safetensors": True
    }
    
    print(f"[CUDA LIBRARY] Backend configured for: {device}")
    print(f"[CUDA LIBRARY] Max memory: {max_memory}")
    print(f"[CUDA LIBRARY] Using 8-bit quantization")
    
    return backend_config

def get_shared_backend():
    """Get or create the shared backend with all heavy libraries pre-loaded"""
    global _shared_backend
    
    if _shared_backend is None:
        print("[CUDA LIBRARY] 🔄 Loading shared backend libraries...")
        
        start_time = time.time()
        
        # Load heavy libraries once
        import torch
        import transformers
        from diffusers import FluxPipeline
        
        _shared_backend = {
            "torch": torch,
            "transformers": transformers,
            "FluxPipeline": FluxPipeline,
            "backend_config": initialize_cuda_backend(),
            "load_time": time.time() - start_time
        }
        
        print(f"[CUDA LIBRARY] ✅ Shared backend loaded in {_shared_backend['load_time']:.2f}s")
        
    return _shared_backend

class HuggingFaceCudaLibraryLoader(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self):
        """Heavy library loading happens here to share across all nodes"""
        print("[CUDA LIBRARY] 🔧 Pre-loading shared backend...")
        
        # Ensure bitsandbytes is ready
        _check_bitsandbytes_cuda_support()
        
        # Pre-load the shared backend
        backend = get_shared_backend()
        print(f"[CUDA LIBRARY] ✅ Shared backend ready: {list(backend.keys())}")
        
    def after_library_nodes_loaded(self):
        """Finalization after all nodes are loaded"""
        print("[CUDA LIBRARY] 🎯 All CUDA nodes loaded successfully")
        
        # Print memory summary if available
        backend = get_shared_backend()
        if "torch" in backend and hasattr(backend["torch"], "cuda") and backend["torch"].cuda.is_available():
            device = backend["torch"].cuda.current_device()
            allocated = backend["torch"].cuda.memory_allocated(device) / 1024**2
            reserved = backend["torch"].cuda.memory_reserved(device) / 1024**2
            print(f"[CUDA LIBRARY] 📊 GPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")

# Export the advanced library class
AdvancedLibrary = HuggingFaceCudaLibraryLoader