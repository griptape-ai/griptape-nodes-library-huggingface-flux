from typing import Dict, Any
import os


def safe_settings_for_env(repo_id: str) -> Dict[str, Any]:
    """Return conservative, HF-aligned defaults for this environment.

    - dtype: bfloat16 on CUDA, float32 otherwise
    - device_map: "balanced" (supported in this env)
    - tokenizer_max_length: 256 for schnell, 512 for others
    """
    try:  # local import to avoid load-time torch requirement
        import torch  # type: ignore
        has_cuda = torch.cuda.is_available()
        # Prefer float16 on CUDA to minimize VRAM usage
        dtype = torch.float16 if has_cuda else torch.float32
    except Exception:
        dtype = None  # pipeline will fall back, but in practice torch is present

    rid = (repo_id or "").lower()
    tok_max = 256 if "schnell" in rid else 512
    # Attention impl decided by library_loader at startup
    attn_impl = os.getenv("GT_ATTENTION_IMPL", "sdpa")
    attn_dbg = os.getenv("GT_ATTENTION_DEBUG", "")

    # Cap total pixels to avoid runaway memory usage; default 1024x1024
    try:
        max_pixels = int(os.getenv("GT_MAX_PIXELS", str(1024 * 1024)))
    except Exception:
        max_pixels = 1024 * 1024

    return {
        "torch_dtype": dtype,
        "device_map": os.getenv("GT_DEVICE_MAP", "balanced"),
        "tokenizer_max_length": tok_max,
        "attn_impl": attn_impl,
        "attn_reason": os.getenv("GT_ATTENTION_REASON", "unknown"),
        "attn_debug": attn_dbg,
        "max_pixels": max_pixels,
    }


