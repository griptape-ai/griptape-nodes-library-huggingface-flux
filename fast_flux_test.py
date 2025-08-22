#!/usr/bin/env python3
"""
Simple maximally optimized FLUX test - hardcoded settings for maximum speed
"""

import time
import torch
import platform
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download
import os

# HARDCODED SETTINGS - EDIT THESE
MODEL_PATH = "XLabs-AI/flux-dev-fp8"
PROMPT = "a photo of a spaceship flying through space"
STEPS = 20
GUIDANCE = 3.5
WIDTH = 1024
HEIGHT = 1024


def main():
    print("🚀 FLUX Maximum Speed Test")
    print("=" * 40)

    # Print system info
    print(f"🖥️  Platform: {platform.system()}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(
            f"🧠 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("⚠️  No CUDA GPU detected")

    # Load model
    print("📥 Loading FLUX...")
    load_start = time.perf_counter()

    try:
        # Try to load as diffusers model first
        pipe = FluxPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    except:
        # If that fails, try loading from single file checkpoint
        try:
            # Download the main checkpoint file
            checkpoint_path = hf_hub_download(
                repo_id=MODEL_PATH,
                filename="flux1-dev-fp8.safetensors",  # Common fp8 filename
                local_dir="./models",
                local_dir_use_symlinks=False
            )
            pipe = FluxPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        except:
            # Last resort - try the base FLUX model
            print("  ⚠️  Using base FLUX.1-dev model instead")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

    print(f"✅ Loaded in {time.perf_counter() - load_start:.2f}s")

    # Apply ALL optimizations
    print("🔧 Applying optimizations...")
    opt_start = time.perf_counter()

    # 1. Quantization
    quant_start = time.perf_counter()
    try:
        from optimum.quanto import freeze, qfloat8, quantize

        quantize(pipe.transformer, weights=qfloat8, exclude=["proj_out"])
        freeze(pipe.transformer)
        print("  ✅ Quantized transformer")

        if hasattr(pipe, "text_encoder_2"):
            quantize(pipe.text_encoder_2, weights=qfloat8)
            freeze(pipe.text_encoder_2)
            print("  ✅ Quantized T5 encoder")
    except:
        print("  ⚠️  Quantization skipped")

    quant_time = time.perf_counter() - quant_start
    print(f"  ⏱️  Quantization took {quant_time:.2f}s")

    # 2. Memory optimizations
    # Skip attention slicing for maximum speed (uses more memory but faster)
    # pipe.enable_attention_slicing()  # Disabled for speed
    # pipe.enable_vae_slicing()        # Disabled for speed

    # Convert to channels_last for better memory access patterns
    pipe.transformer = pipe.transformer.to(memory_format=torch.channels_last)
    if hasattr(pipe, "vae"):
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

    # Enable QKV fusion if available
    try:
        if hasattr(pipe.transformer, "fuse_qkv_projections"):
            pipe.transformer.fuse_qkv_projections()
            print("  ✅ QKV fusion enabled")
    except:
        pass

    print("  ✅ Memory optimizations applied")

    # 3. Device setup - Keep everything on GPU for speed
    device_start = time.perf_counter()
    if torch.cuda.is_available():
        pipe.to("cuda")  # Keep entire pipeline on GPU
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable flash attention if available
        try:
            torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            )
        except:
            pass
        print("  ✅ GPU optimizations enabled")

    device_time = time.perf_counter() - device_start
    print(f"  ⏱️  Device setup took {device_time:.2f}s")

    # 4. Compile models
    compile_start = time.perf_counter()
    try:
        torch._dynamo.config.suppress_errors = True

        if platform.system() == "Windows":
            # Windows safe compilation
            torch._dynamo.config.assume_static_by_default = True
            pipe.transformer = torch.compile(
                pipe.transformer, mode="reduce-overhead", backend="eager"
            )
            print("  ✅ Compiled for Windows")
        else:
            # Full optimization for Linux/macOS
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = True

            pipe.transformer = torch.compile(
                pipe.transformer, mode="max-autotune", fullgraph=True
            )
            pipe.vae = torch.compile(pipe.vae, mode="max-autotune", fullgraph=True)
            print("  ✅ Compiled with full optimization")
    except:
        print("  ⚠️  Compilation skipped")

    compile_time = time.perf_counter() - compile_start
    print(f"  ⏱️  Compilation took {compile_time:.2f}s")

    total_opt_time = time.perf_counter() - opt_start
    print(f"🔧 All optimizations applied in {total_opt_time:.2f}s")

    # Warmup - use smaller size first
    print("🔥 Warmup...")
    warmup_start = time.perf_counter()
    with torch.inference_mode():
        _ = pipe("test", num_inference_steps=1, height=256, width=256)
    warmup_time = time.perf_counter() - warmup_start
    print(f"  ⏱️  Warmup took {warmup_time:.2f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Main generation
    print(f"🎨 Generating: '{PROMPT}'")
    print(f"Settings: {STEPS} steps, {GUIDANCE} guidance, {WIDTH}x{HEIGHT}")

    start = time.perf_counter()

    with torch.inference_mode():
        result = pipe(
            PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            height=HEIGHT,
            width=WIDTH,
        )
        result = pipe(
            PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            height=HEIGHT,
            width=WIDTH,
        )

    total_time = time.perf_counter() - start

    # Results
    print("\n📊 RESULTS:")
    print(f"⚡ Time: {total_time:.2f}s")
    print(f"⚡ Speed: {STEPS / total_time:.2f} steps/sec")

    if torch.cuda.is_available():
        print(f"🧠 GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Save
    output_file = f"fast_flux_{int(time.time())}.png"
    result.images[0].save(output_file)
    print(f"💾 Saved: {output_file}")

    print(f"\n🎉 DONE! Generated in {total_time:.2f} seconds")


if __name__ == "__main__":
    main()

