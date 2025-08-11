import os
import sys
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
        self._device, self._dtype, self._attn, self._attn_reason, self._attn_debug = select_device_dtype_attn()
        # Persist attention decision for the rest of the session so nodes/runners
        # can read it without re-probing and risking runtime failures.
        os.environ.setdefault("GT_ATTENTION_IMPL", self._attn)
        os.environ.setdefault("GT_ATTENTION_REASON", self._attn_reason)
        os.environ.setdefault("GT_ATTENTION_DEBUG", self._attn_debug)
        try:
            import logging
            logging.getLogger(__name__).warning(
                f"[HF-Diffusers] attention_impl={self._attn} reason={self._attn_reason} debug=({self._attn_debug})"
            )
        except Exception:
            pass

    def after_library_nodes_loaded(self, library_data=None, library=None):
        pass

def select_device_dtype_attn():
    """Lazy torch import so manifest load doesn't require torch preinstalled."""
    try:
        import torch  # noqa: F401
    except Exception:
        return "cpu", None, "sdpa", "no-torch", "platform=?, python=?, torch=?, cuda=?, cc=?, fa2=missing"
    import torch as _torch
    if _torch.cuda.is_available():
        major, _ = _torch.cuda.get_device_capability()
        dtype = _torch.bfloat16 if major >= 8 else _torch.float16
        device = "cuda"

        # Decide attention implementation once at load time.
        attn = "sdpa"
        reason = "sdpa-default"
        fa2_ver = "missing"
        if sys.platform == "linux" and major >= 8:
            try:
                import importlib
                fa2 = importlib.import_module("flash_attn")
                ver = getattr(fa2, "__version__", "0.0.0")
                parts = tuple(int(x) for x in ver.split(".")[:3] if x.isdigit())
                if parts >= (2, 5, 9):
                    attn = "flash_attention_2"
                    reason = f"flash_attn=={ver}"
                    fa2_ver = ver
                else:
                    reason = f"flash_attn-too-old=={ver}"
                    fa2_ver = ver
            except Exception:
                attn = "sdpa"
                reason = "flash_attn-missing"
        debug = (
            f"platform={sys.platform}, python={sys.version.split()[0]}, "
            f"torch={_torch.__version__}, cuda={getattr(_torch.version, 'cuda', '?')}, "
            f"cc={major}, fa2={fa2_ver}"
        )
        return device, dtype, attn, reason, debug
    if getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
        debug = (
            f"platform={sys.platform}, python={sys.version.split()[0]}, torch={_torch.__version__}, mps=available"
        )
        return "mps", _torch.float16, "sdpa", "mps", debug
    debug = (
        f"platform={sys.platform}, python={sys.version.split()[0]}, torch={_torch.__version__}, cpu=true"
    )
    return "cpu", _torch.float32, "sdpa", "cpu", debug

# Export symbol expected by the engine
AdvancedLibrary = Library
