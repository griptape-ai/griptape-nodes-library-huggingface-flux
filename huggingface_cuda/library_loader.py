import logging
import time

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Verify this module is being imported
print("[CUDA LIBRARY] 📦 Library loader module imported successfully")

# Global shared backend storage
_shared_backend = None

def _check_bitsandbytes_cuda_support():
    """Check if bitsandbytes has CUDA support and provide guidance if not"""
    try:
        import bitsandbytes as bnb
        
        # Try to check CUDA support
        try:
            # This will work if CUDA support is available
            import bitsandbytes.cuda_setup
            print("[CUDA LIBRARY] ✅ bitsandbytes CUDA support detected")
        except (ImportError, AttributeError):
            # Fallback: try to create a CUDA quantization config
            try:
                from transformers import BitsAndBytesConfig
                import torch
                if torch.cuda.is_available():
                    # Test if we can create a quantization config
                    config = BitsAndBytesConfig(load_in_4bit=True)
                    print("[CUDA LIBRARY] ✅ bitsandbytes quantization available")
                else:
                    print("[CUDA LIBRARY] ⚠️ CUDA not available, quantization will use CPU")
            except Exception as e:
                print(f"[CUDA LIBRARY] ⚠️ bitsandbytes CUDA issue detected: {e}")
                print("[CUDA LIBRARY] 💡 For RTX 4090 quantization support, try:")
                print("[CUDA LIBRARY]    pip uninstall bitsandbytes")
                print("[CUDA LIBRARY]    pip install bitsandbytes --upgrade --force-reinstall --no-cache-dir")
                print("[CUDA LIBRARY]    Or visit: https://github.com/bitsandbytes-foundation/bitsandbytes")
                
    except ImportError:
        print("[CUDA LIBRARY] ⚠️ bitsandbytes not available - quantization disabled")

def initialize_cuda_backend():
    """
    Initialize CUDA backend with heavy imports.
    Called before nodes are loaded to optimize node creation time.
    """
    global _shared_backend
    
    if _shared_backend is not None:
        print("[CUDA LIBRARY] Backend already initialized")
        return
    
    print("[CUDA LIBRARY] 🚀 Initializing CUDA backend via library loader...")
    start_time = time.time()
    
    try:
        # Heavy imports - do these once at library load time
        print("[CUDA LIBRARY] Importing PyTorch...")
        import torch
        
        print("[CUDA LIBRARY] Importing Diffusers...")
        from diffusers import FluxPipeline
        
        print("[CUDA LIBRARY] Importing Transformers...")
        from transformers import CLIPTextModel, T5EncoderModel, BitsAndBytesConfig
        
        print("[CUDA LIBRARY] Importing NumPy...")
        import numpy as np
        
        # Pre-initialize device info
        device_info = None
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_info = {
                'device_count': device_count,
                'devices': []
            }
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                device_info['devices'].append({
                    'id': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'major': props.major,
                    'minor': props.minor
                })
            print(f"[CUDA LIBRARY] Detected {device_count} CUDA device(s)")
        else:
            print("[CUDA LIBRARY] CUDA not available, CPU fallback will be used")
            # Add detailed CUDA diagnostics
            print(f"[CUDA LIBRARY] 🔍 CUDA Diagnostics:")
            print(f"[CUDA LIBRARY]   PyTorch version: {torch.__version__}")
            print(f"[CUDA LIBRARY]   PyTorch CUDA compiled version: {torch.version.cuda}")
            print(f"[CUDA LIBRARY]   PyTorch CUDA available: {torch.cuda.is_available()}")
            
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("[CUDA LIBRARY]   nvidia-smi: GPU detected by system")
                    # Extract GPU info from nvidia-smi
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'RTX' in line or 'GTX' in line or 'Tesla' in line:
                            print(f"[CUDA LIBRARY]   System GPU: {line.strip()}")
                            break
                else:
                    print("[CUDA LIBRARY]   nvidia-smi: No GPU detected by system")
            except Exception as e:
                print(f"[CUDA LIBRARY]   nvidia-smi check failed: {e}")
            
            print("[CUDA LIBRARY] 💡 For RTX 4090 CUDA support:")
            print("[CUDA LIBRARY]    1. Check NVIDIA drivers are installed")
            print("[CUDA LIBRARY]    2. PyTorch may need CUDA-enabled version:")
            print("[CUDA LIBRARY]       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            print("[CUDA LIBRARY]    3. Restart Griptape after PyTorch reinstall")
        
        # Store in global backend
        _shared_backend = {
            'torch': torch,
            'FluxPipeline': FluxPipeline,
            'CLIPTextModel': CLIPTextModel,
            'T5EncoderModel': T5EncoderModel,
            'BitsAndBytesConfig': BitsAndBytesConfig,
            'numpy': np,
            'device_info': device_info,
            'loaded_at': time.time(),
            'available': True
        }
        
        load_time = time.time() - start_time
        print(f"[CUDA LIBRARY] ✅ Backend initialized successfully in {load_time:.2f}s")
        
        # Check bitsandbytes CUDA support
        _check_bitsandbytes_cuda_support()
        
    except ImportError as e:
        print(f"[CUDA LIBRARY] Failed to initialize backend: {e}")
        _shared_backend = {
            'available': False,
            'error': str(e),
            'loaded_at': time.time()
        }
    except Exception as e:
        print(f"[CUDA LIBRARY] Unexpected error during initialization: {e}")
        _shared_backend = {
            'available': False,
            'error': str(e),
            'loaded_at': time.time()
        }

def get_shared_backend():
    """
    Get the pre-loaded shared backend. Should already be initialized.
    Returns dict with pre-loaded modules and device info.
    """
    global _shared_backend
    
    if _shared_backend is None:
        print("[CUDA LIBRARY] ⚠️ Backend not pre-loaded! Initializing on-demand (this will be slow)...")
        initialize_cuda_backend()
    
    return _shared_backend


class HuggingFaceCudaLibraryLoader(AdvancedNodeLibrary):
    """
    Library loader for the HuggingFace CUDA Library.
    
    This class implements Griptape's AdvancedNodeLibrary pattern to pre-load
    heavy dependencies (PyTorch, Diffusers, Transformers) before any nodes are
    created, resulting in fast node initialization times.
    
    Key responsibilities:
    - Pre-load PyTorch and check CUDA availability  
    - Pre-load Diffusers FluxPipeline
    - Pre-load Transformers models and quantization configs
    - Create shared backend accessible to all nodes
    - Provide detailed diagnostics for troubleshooting
    """
    
    def __init__(self):
        """Initialize the library loader."""
        print("[CUDA LIBRARY] 🎯 Library loader class instantiated successfully")
        super().__init__()

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library):
        """
        Called before any nodes are loaded from the library.
        
        This is where we do all the heavy lifting - importing large ML libraries
        and initializing the shared backend. By doing this once here instead of
        in each node's __init__, we get:
        
        - Fast node creation (< 500ms instead of 10+ seconds)
        - Shared state across all nodes
        - Better memory management
        - Cleaner error handling
        """
        print(f"[CUDA LIBRARY] 🚀 Starting to load nodes for '{library_data.name}' library...")
        
        # Initialize the CUDA backend before any nodes are created
        initialize_cuda_backend()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library):
        """
        Called after all nodes have been loaded from the library.
        
        This is where we verify everything loaded correctly and provide
        useful status information to the user.
        """
        backend = get_shared_backend()
        if backend and backend.get('available', False):
            load_time = time.time() - backend['loaded_at']
            print(f"[CUDA LIBRARY] ✅ All nodes loaded successfully. Backend ready in {load_time:.2f}s")
            if 'device_info' in backend and backend['device_info']:
                device_count = backend['device_info']['device_count']
                print(f"[CUDA LIBRARY] 🎯 Ready for fast inference on {device_count} CUDA device(s)")
            else:
                print(f"[CUDA LIBRARY] 🎯 Ready for CPU inference")
        else:
            error = backend.get('error', 'Unknown error') if backend else 'Backend not available'
            print(f"[CUDA LIBRARY] ⚠️ Library loaded but backend has issues: {error}")


# Export the class for Griptape's AdvancedNodeLibrary discovery
# Note: This must be named "AdvancedLibrary" for Griptape to find it
AdvancedLibrary = HuggingFaceCudaLibraryLoader 