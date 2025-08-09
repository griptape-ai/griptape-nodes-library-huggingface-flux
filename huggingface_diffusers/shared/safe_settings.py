from typing import Dict, Any


def safe_settings_for_env(repo_id: str) -> Dict[str, Any]:
    """Return conservative, HF-aligned defaults for this environment.

    - dtype: bfloat16 on CUDA, float32 otherwise
    - device_map: "balanced" (supported in this env)
    - tokenizer_max_length: 256 for schnell, 512 for others
    """
    try:  # local import to avoid load-time torch requirement
        import torch  # type: ignore
        has_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if has_cuda else torch.float32
    except Exception:
        dtype = None  # pipeline will fall back, but in practice torch is present

    rid = (repo_id or "").lower()
    tok_max = 256 if "schnell" in rid else 512

    return {
        "torch_dtype": dtype,
        "device_map": "balanced",
        "tokenizer_max_length": tok_max,
    }


