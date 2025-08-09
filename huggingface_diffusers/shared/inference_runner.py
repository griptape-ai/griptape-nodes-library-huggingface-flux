from typing import Any, List, Tuple, Dict
import time
from io import BytesIO

from diffusers import FluxPipeline
from collections import OrderedDict

try:
    from .safe_settings import safe_settings_for_env  # type: ignore
except Exception:
    # Fallback for loose-module load
    from huggingface_diffusers.shared.safe_settings import safe_settings_for_env  # type: ignore


# Minimal in-process cache: reuse the same pipeline for repeated runs of the SAME local_path only
_PIPELINE_CACHE: "OrderedDict[str, FluxPipeline]" = OrderedDict()
_PIPELINE_CACHE_MAX = 3


def run_flux_inference(
    local_path: str,
    repo_id: str,
    prompt: str,
    negative_prompt: str | None,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_images_per_prompt: int = 1,
) -> Tuple[List[bytes], dict]:
    """Load pipeline with safe settings and run a single inference call.

    Returns (list of PNG bytes, timings dict)
    """
    s = safe_settings_for_env(repo_id)
    t0 = time.perf_counter()
    pipe = _PIPELINE_CACHE.get(local_path)
    if pipe is None:
        pipe = FluxPipeline.from_pretrained(
            local_path,
            torch_dtype=s.get("torch_dtype"),
            local_files_only=True,
        )
        # Standard HF offload for stability – lets Diffusers manage device moves
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload()
        except Exception:
            pass
        _PIPELINE_CACHE[local_path] = pipe
        # Enforce small LRU
        while len(_PIPELINE_CACHE) > _PIPELINE_CACHE_MAX:
            _PIPELINE_CACHE.popitem(last=False)
    else:
        # LRU bump on reuse
        _PIPELINE_CACHE.move_to_end(local_path, last=True)
    t_load = time.perf_counter() - t0

    # Tokenizer length (best-effort)
    try:
        pipe.tokenizer_2.model_max_length = int(s.get("tokenizer_max_length", 512))
    except Exception:
        pass

    # Inference
    t1 = time.perf_counter()
    import torch  # type: ignore
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            height=int(height),
            width=int(width),
            num_images_per_prompt=int(max(1, min(4, num_images_per_prompt))),
        ).images
    t_infer = time.perf_counter() - t1

    # Encode to PNG bytes
    t2 = time.perf_counter()
    pngs: List[bytes] = []
    for pil_img in out:
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        pngs.append(buf.getvalue())
    t_save = time.perf_counter() - t2

    # Timings + settings summary for performance report
    try:
        dtype_str = str(s.get("torch_dtype"))
    except Exception:
        dtype_str = str(None)
    timings = {
        "load_pipe_s": t_load,
        "infer_s": t_infer,
        "encode_s": t_save,
        "total_s": time.perf_counter() - t0,
        "cache_entries": len(_PIPELINE_CACHE),
        "torch_dtype": dtype_str,
        "tokenizer_max_length": int(s.get("tokenizer_max_length", 512)),
        "num_images_per_prompt": int(max(1, min(4, num_images_per_prompt))),
    }
    return pngs, timings


