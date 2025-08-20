import os
import sys
import platform
import subprocess
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
        # If a pinned torch wheel URL is provided via settings, install it first
        try:
            self._maybe_install_torch_from_wheel(library_data)
        except Exception:
            pass
        # Opportunistically upgrade torch to a CUDA build on Windows if CPU-only was installed by resolver
        try:
            self._maybe_upgrade_torch_cuda()
        except Exception:
            pass
        # Defer heavy imports until dependencies are installed/upgraded in the engine sandbox
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

    def _maybe_install_torch_from_wheel(self, library_data=None) -> None:
        """Install a pinned torch wheel URL if provided in settings.

        Setting key: GT_TORCH_WHEEL_URL (or TORCH_WHEEL_URL). Uses --no-deps to avoid resolver conflicts.
        """
        # Read from settings first
        url = None
        try:
            settings = (library_data or {}).get("settings", [])
            for s in settings:
                contents = s.get("contents", {})
                url = contents.get("GT_TORCH_WHEEL_URL") or contents.get("TORCH_WHEEL_URL")
                if url:
                    break
        except Exception:
            url = None
        # Allow env override
        url = os.getenv("GT_TORCH_WHEEL_URL", url)
        if not url:
            return
        py = sys.executable
        cmd = [py, "-m", "pip", "install", "--upgrade", "--no-deps", url]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([py, "-c", "import torch"], check=False)

    def _maybe_upgrade_torch_cuda(self) -> None:
        """On Windows + NVIDIA, ensure CUDA torch is present. If not, try nightly cu126 upgrade.

        This runs before we import torch elsewhere. If upgrade succeeds, subsequent imports
        will see the CUDA build. If it fails, we silently continue and default to CPU.
        """
        if sys.platform != "win32":
            return
        # Respect opt-out
        if os.getenv("GT_SKIP_TORCH_CUDA_UPGRADE", "").lower() in ("1", "true", "yes"):
            return
        # Quick probe without importing torch first (to avoid locking CPU wheel in memory)
        py = sys.executable
        probe = subprocess.run([py, "-c", "import torch; import json; print(json.dumps({'cuda': bool(getattr(torch, 'cuda', None) and torch.cuda.is_available()), 'ver': getattr(torch, '__version__', 'unknown'), 'cudaver': getattr(getattr(torch, 'version', None), 'cuda', None)}))"],
                               capture_output=True, text=True)
        cuda_ok = False
        if probe.returncode == 0:
            try:
                import json as _json
                info = _json.loads(probe.stdout.strip())
                cuda_ok = bool(info.get("cuda"))
            except Exception:
                pass
        if cuda_ok:
            return
        # Try upgrade to CUDA nightly (cu126). Keep this minimal and quiet.
        cmd = [
            py,
            "-m", "pip", "install", "--upgrade", "--pre",
            "--index-url", "https://download.pytorch.org/whl/nightly/cu126",
            "torch",
        ]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Second probe; if still not CUDA, leave as-is.
        subprocess.run([py, "-c", "import torch"], check=False)

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
