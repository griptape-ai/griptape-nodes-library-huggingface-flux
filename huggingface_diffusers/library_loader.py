import os
import platform
from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary

class Library(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data=None, library=None):
        # Decide backend path; for first cut force Diffusers SDPA everywhere
        os.environ.setdefault("HF_HUB_CACHE", os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
        # Propagate HF token from settings if provided (matches bkup behavior)
        try:
            settings = (library_data or {}).get("settings", [])
            for s in settings:
                contents = s.get("contents", {})
                token = contents.get("HUGGINGFACE_HUB_ACCESS_TOKEN") or contents.get("HF_TOKEN")
                if token:
                    os.environ.setdefault("HUGGINGFACE_HUB_ACCESS_TOKEN", token)
                    os.environ.setdefault("HF_TOKEN", token)
                    break
        except Exception:
            pass
        # Defer heavy imports until dependencies are installed in the engine sandbox
        self._device, self._dtype, self._attn = select_device_dtype_attn()

    def after_library_nodes_loaded(self, library_data=None, library=None):
        pass

def select_device_dtype_attn():
    """Lazy torch import so manifest load doesn't require torch preinstalled."""
    try:
        import torch  # noqa: F401
    except Exception:
        return "cpu", None, "sdpa"
    import torch as _torch
    if _torch.cuda.is_available():
        major, _ = _torch.cuda.get_device_capability()
        dtype = _torch.bfloat16 if major >= 8 else _torch.float16
        attn = "sdpa"
        device = "cuda"
        return device, dtype, attn
    if getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
        return "mps", _torch.float16, "sdpa"
    return "cpu", _torch.float32, "sdpa"

# Export symbol expected by the engine
AdvancedLibrary = Library
