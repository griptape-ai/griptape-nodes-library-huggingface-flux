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
    device_policy: str | None = None,  # "auto" | "gpu" | "cpu_offload"
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
        # Device/offload policy: on large VRAM boxes, keep everything on GPU for speed
        try:
            import os
            import torch  # type: ignore
            if torch.cuda.is_available():
                total_mem = getattr(torch.cuda.get_device_properties(0), "total_memory", 0)
                policy = (device_policy or "auto").lower()
                if policy not in ("auto", "gpu", "cpu_offload"):
                    policy = "auto"
                # default auto: if VRAM < 24GB, enable offload; else keep on GPU
                enable_offload = (policy == "cpu_offload") or (policy == "auto" and total_mem < (24 * 1024**3))
                if enable_offload:
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to("cuda")
                # Prefer flash SDP kernels in PyTorch when available
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends.cuda, "sdp_kernel"):
                        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
                except Exception:
                    pass
        except Exception:
            pass
        # Apply attention implementation chosen at library load
        try:
            attn_impl = s.get("attn_impl", "sdpa")
            if attn_impl == "flash_attention_2" and hasattr(pipe, "transformer") and hasattr(pipe.transformer, "set_attn_implementation"):
                pipe.transformer.set_attn_implementation("flash_attention_2")
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

    # Inference with OOM fallbacks
    import torch  # type: ignore
    used_policy = device_policy or "auto"
    oom_retries = 0
    reduced_batch = False
    used_cpu_preencode = False
    def _do_call(n_imgs: int):
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            height=int(height),
            width=int(width),
            num_images_per_prompt=int(max(1, min(4, n_imgs))),
        ).images

    t1 = time.perf_counter()
    try:
        with torch.inference_mode():
            out = _do_call(num_images_per_prompt)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            oom_retries += 1
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            # First fallback: enable offload if currently on full GPU
            try:
                if torch.cuda.is_available():
                    if hasattr(pipe, "enable_model_cpu_offload"):
                        pipe.enable_model_cpu_offload()
                        used_policy = "cpu_offload"
                        with torch.inference_mode():
                            out = _do_call(num_images_per_prompt)
                    else:
                        raise e
                else:
                    raise e
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                # Second fallback: enable sequential offload for lowest peak VRAM
                try:
                    if hasattr(pipe, "enable_sequential_cpu_offload"):
                        pipe.enable_sequential_cpu_offload()
                        used_policy = "sequential_offload"
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        with torch.inference_mode():
                            out = _do_call(num_images_per_prompt)
                    else:
                        raise
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    # Third fallback: run text encoders on CPU (float32) to free GPU VRAM
                    try:
                        moved_any = False
                        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                            try:
                                pipe.text_encoder.to("cpu")
                                moved_any = True
                            except Exception:
                                pass
                        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
                            try:
                                pipe.text_encoder_2.to("cpu")
                                moved_any = True
                            except Exception:
                                pass
                        if moved_any:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            with torch.inference_mode():
                                out = _do_call(num_images_per_prompt)
                        else:
                            raise
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        # Fourth fallback: pre-encode text on CPU and pass embeddings to pipeline
                        try:
                            from torch import nn  # type: ignore
                            # Ensure tokenizers exist
                            tok = getattr(pipe, "tokenizer", None)
                            tok2 = getattr(pipe, "tokenizer_2", None)
                            te = getattr(pipe, "text_encoder", None)
                            te2 = getattr(pipe, "text_encoder_2", None)
                            if tok is None or tok2 is None or te is None or te2 is None:
                                raise RuntimeError("Missing tokenizer/encoder components for CPU pre-encode")
                            # Build inputs
                            prompts = [prompt] * int(max(1, num_images_per_prompt))
                            negs = [negative_prompt] * int(max(1, num_images_per_prompt)) if negative_prompt else None
                            # CLIP pooled on CPU
                            clip_inputs = tok(prompts, padding=True, truncation=True, return_tensors="pt")
                            clip_neg_inputs = tok(negs, padding=True, truncation=True, return_tensors="pt") if negs else None
                            te = te.to("cpu")
                            te.eval()
                            with torch.no_grad():
                                pooled = te(**{k: v.to("cpu") for k, v in clip_inputs.items()})
                                pooled = pooled.pooler_output
                                pooled_neg = None
                                if clip_neg_inputs is not None:
                                    pooled_neg = te(**{k: v.to("cpu") for k, v in clip_neg_inputs.items()}).pooler_output
                            # T5 last_hidden_state on CPU
                            t5_inputs = tok2(prompts, padding=True, truncation=True, return_tensors="pt")
                            t5_neg_inputs = tok2(negs, padding=True, truncation=True, return_tensors="pt") if negs else None
                            te2 = te2.to("cpu")
                            te2.eval()
                            with torch.no_grad():
                                text_embeds = te2(**{k: v.to("cpu") for k, v in t5_inputs.items()}).last_hidden_state
                                text_embeds_neg = None
                                if t5_neg_inputs is not None:
                                    text_embeds_neg = te2(**{k: v.to("cpu") for k, v in t5_neg_inputs.items()}).last_hidden_state
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            used_cpu_preencode = True
                            # Call pipeline bypassing internal encoding
                            with torch.inference_mode():
                                out = pipe(
                                    prompt=None,
                                    negative_prompt=None,
                                    prompt_embeds=text_embeds,
                                    pooled_prompt_embeds=pooled,
                                    negative_prompt_embeds=text_embeds_neg,
                                    negative_pooled_prompt_embeds=pooled_neg,
                                    num_inference_steps=int(num_inference_steps),
                                    guidance_scale=float(guidance_scale),
                                    height=int(height),
                                    width=int(width),
                                    num_images_per_prompt=int(max(1, min(4, num_images_per_prompt))),
                                ).images
                        except (torch.cuda.OutOfMemoryError, RuntimeError):
                            # Fifth fallback: disable FA2 (use SDPA) and try again
                            try:
                                if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "set_attn_implementation"):
                                    pipe.transformer.set_attn_implementation("sdpa")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                with torch.inference_mode():
                                    out = _do_call(num_images_per_prompt)
                            except (torch.cuda.OutOfMemoryError, RuntimeError):
                                # Sixth fallback: reduce batch to 1
                                oom_retries += 1
                                reduced_batch = True
                                with torch.inference_mode():
                                    out = _do_call(1)
        else:
            raise
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
        "attn_impl": s.get("attn_impl", "sdpa"),
        "attn_reason": s.get("attn_reason", "unknown"),
        "attn_debug": s.get("attn_debug", ""),
        "device_policy": (device_policy or "auto"),
        "used_device_policy": used_policy,
        "oom_retries": oom_retries,
        "reduced_batch_to_1": reduced_batch,
        "cpu_preencode": used_cpu_preencode,
    }
    return pngs, timings


