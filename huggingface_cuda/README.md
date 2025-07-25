# HuggingFace CUDA Library

A FLUX inference library for NVIDIA CUDA and CPU systems with intelligent model discovery, advanced quantization, and optimized memory management.

## Current Status: **In Development**

This library provides FLUX model inference with automatic CUDA detection and seamless CPU fallback. The architecture uses an advanced library loader pattern for fast node initialization.

## Features

### Dynamic Model Discovery
- Automatic HuggingFace cache scanning to discover locally downloaded FLUX models
- Smart filtering to exclude LoRA, ControlNet, and encoder-only repositories  
- Structure analysis that validates models by examining `model_index.json` and `config.json`
- Real-time updates when new models are downloaded

### Advanced Quantization Support
- **4-bit Quantization**: Pure GPU inference (~8-12GB VRAM, minimal quality loss)
- **8-bit Quantization**: Hybrid GPU/CPU with automatic offload (~15GB GPU + system RAM)
- **Full Precision**: Maximum quality with intelligent CPU offload (25GB+ total memory)
- Automatic memory management with dynamic device mapping and memory limits

### Performance Optimizations
- Shared backend system with pre-loaded PyTorch, Diffusers, and Transformers
- Fast node initialization (<500ms load times via `library_loader.py`)
- Memory efficient operation with intelligent CPU offload when GPU memory insufficient
- Automatic RTX 4090 and multi-GPU support

### System Intelligence
- CUDA auto-detection with automatic fallback to CPU when CUDA unavailable
- Comprehensive diagnostics and detailed troubleshooting information
- Real-time status updates during generation
- Graceful error handling for memory and configuration issues

## Architecture

```
huggingface_cuda/
├── library_loader.py          # Advanced library with pre-loading
├── griptape-nodes-library.json # Library configuration
├── __init__.py                 # Public API
├── flux/
│   └── flux_inference.py       # Main FLUX inference node
└── shared/                     # Shared utilities
    ├── gpu_utils.py            # GPU detection and management
    └── text_encoders.py        # Text encoder utilities
```

## Model Discovery Process

The library uses a sophisticated model discovery system:

### 1. Cache Scanning
```python
from huggingface_hub import scan_cache_dir
# Scans ~/.cache/huggingface/hub for downloaded models
```

### 2. Pattern Filtering
- **Include**: `flux`, `black-forest-labs/FLUX.*`
- **Exclude**: `*lora*`, `*controlnet*`, `*text_encoders*`, `*clip*`, `*t5*`

### 3. Structure Validation
- Checks for `model_index.json` with `FluxPipeline` class
- Validates presence of core components: `transformer`, `scheduler`, `vae`
- Excludes ControlNet and specialized models

### 4. Fallback Handling
- Uses default models if scanning fails
- Provides clear error messages for troubleshooting

## Quantization Options

### 4-bit Quantization (Recommended)
- **Memory**: ~8-12GB GPU only
- **Speed**: Fastest
- **Quality**: Excellent (minimal loss)
- **Best for**: Most users, fast iteration

### 8-bit Quantization (Quality Focus)
- **Memory**: ~15GB GPU + 5-10GB system RAM  
- **Speed**: Good (CPU offload overhead)
- **Quality**: Excellent (minimal loss)
- **Best for**: Quality-critical work, sufficient system RAM (16GB+)

### Full Precision (Maximum Quality)
- **Memory**: 20GB+ GPU + 10-15GB+ system RAM
- **Speed**: Slowest (heavy CPU offload)
- **Quality**: Maximum
- **Best for**: Research, large system RAM (32GB+)

## Supported Models

Currently supports FLUX text-to-image models:
- **black-forest-labs/FLUX.1-dev** (20 steps recommended)
- **black-forest-labs/FLUX.1-schnell** (4 steps optimized)
- **Any FLUX-compatible model** in HuggingFace format

Models are auto-discovered when downloaded via HuggingFace Model Download node or manual download.

### Optional Dependencies
- `bitsandbytes` for quantization (auto-installed)
- `accelerate` for optimized loading (auto-installed)

## Troubleshooting

### Model Dropdown Empty
1. Download FLUX models via HuggingFace Model Download node
2. Check cache: `~/.cache/huggingface/hub`
3. Restart Griptape to refresh model scan

### CUDA Not Detected
1. Verify NVIDIA drivers are installed and up to date
2. Delete the library's `.venv` directory to force Griptape to recreate it with correct dependencies
3. Restart Griptape Nodes to reinstall dependencies from `griptape-nodes-library.json`
4. Check that `griptape-nodes-library.json` contains correct CUDA PyTorch dependencies and `--extra-index-url https://download.pytorch.org/whl/cu124`

### Quantization Failures
1. 8-bit issues → Try 4-bit quantization
2. Memory errors → Increase system RAM or use CPU offload
3. Check bitsandbytes CUDA support


## Contributing

This library follows the Griptape Nodes Library standards. See the root README for contribution guidelines.

---
