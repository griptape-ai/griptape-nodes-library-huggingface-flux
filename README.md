# Griptape Nodes: HuggingFace FLUX Library

A Griptape Nodes library for running FLUX text-to-image models locally with optimized backends.

## 🚀 **What This Library Provides**

- **FLUX.1-dev** and **FLUX.1-schnell** support
- **Apple Silicon (MLX)** optimized inference 
- **Dynamic model discovery** from HuggingFace cache
- **Custom T5/CLIP encoders** with safetensors support
- **Quantization options** (4-bit, 8-bit, none) for memory efficiency
- **Pre-quantized model detection** and automatic configuration

## 📋 **Current Status**

| Backend | Status | Notes |
|---------|--------|-------|
| **MLX (Apple Silicon)** | ✅ **Working** | M1/M2/M3 Macs, work in progress |
| **CUDA (NVIDIA)** | ❌ **Not Working** | Coming in future release |

## 🛠 **Requirements**

### System Requirements
- **Apple Silicon Mac** (M1/M2/M3) for MLX backend
- **macOS** with Python 3.9+
- **HuggingFace account** for model downloads

### Dependencies
The library automatically installs:
- `mlx>=0.19.0` - Apple's MLX framework
- `mlx-lm>=0.18.0` - MLX language model support
- `git+https://github.com/griptape-ai/mflux.git@encoder-support` - Custom FLUX inference (our fork)
- `transformers>=4.53.2` - HuggingFace transformers
- `huggingface_hub>=0.24.0` - HuggingFace model hub

## 📦 **Installation**

### 1. Download the Library
```bash
# Navigate to your Griptape Nodes workspace
cd $(gtn config | grep workspace_directory | cut -d'"' -f4)

# Clone this repository
git clone https://github.com/griptape-ai/griptape-nodes-library-huggingface-flux.git
```

### 2. Register the Library
1. **Copy the JSON path**: Right-click on `huggingface_flux_mlx/griptape-nodes-library.json` and select "Copy Path"
2. **Open Griptape Nodes** and navigate to Settings
3. **Go to App Events tab** → **Libraries to Register** → **Add Item**
4. **Paste the absolute path** to `huggingface_flux_mlx/griptape-nodes-library.json`
5. **Restart Griptape Nodes** - the library will appear in the sidebar

> **⚠️ Important**: Use the JSON file from the **`huggingface_flux_mlx/`** subdirectory, not the root directory.

### 3. Download Your First Model
```bash
# Install HuggingFace CLI if needed
pip install huggingface_hub

# Download FLUX.1-schnell (faster, 4 steps)
huggingface-cli download black-forest-labs/FLUX.1-schnell

# Or download FLUX.1-dev (higher quality, 15-50 steps)
huggingface-cli download black-forest-labs/FLUX.1-dev
```

## 🔍 **How Model Discovery Works**

The library **automatically scans your HuggingFace cache** and discovers:

### **FLUX Models**
- Searches `~/.cache/huggingface/hub/` for repositories matching FLUX patterns
- Filters out ControlNets, LoRAs, and encoder-only repositories
- Analyzes model structure (checks for transformer, VAE, scheduler components)

### **T5 Text Encoders**
- Finds T5 models: `google/t5-*`, `google/flan-t5-*`
- Discovers FLUX-specific encoders: `comfyanonymous/flux_text_encoders`
- Lists individual `.safetensors` files for granular selection
- Filters out unsupported `.bin` formats (MLX requires safetensors)

### **CLIP Text Encoders**
- Finds CLIP models: `openai/clip-*`, `laion/clip-*`
- Discovers FLUX-specific encoders from encoder repositories
- Lists individual `.safetensors` files
- Provides "None (use model default)" option

## ⚙️ **Configuration**

### Customize Model Settings
Edit `huggingface_flux_mlx/flux_config.json` to customize defaults:

```json
{
  "global_defaults": {
    "default_steps": 15,
    "default_quantization": "4-bit",
    "default_width": 1024,
    "default_height": 1024,
    "default_guidance": 7.5,
    "quantization_options": ["none", "4-bit", "8-bit"],
    "default_t5_encoder": "None (use model default)",
    "default_clip_encoder": "None (use model default)"
  },
  "models": {
    "black-forest-labs/FLUX.1-dev": {
      "display_name": "FLUX.1 Dev",
      "mflux_name": "dev",
      "default_steps": 15,
      "max_steps": 50,
      "supports_guidance": true,
      "recommended_t5_encoder": "None (use model default)",
      "recommended_clip_encoder": "None (use model default)"
    },
    "black-forest-labs/FLUX.1-schnell": {
      "display_name": "FLUX.1 Schnell", 
      "mflux_name": "schnell",
      "default_steps": 4,
      "max_steps": 8,
      "supports_guidance": false,
      "default_guidance": 1.0
    },
    "Kijai/flux-fp8": {
      "display_name": "FLUX.1 Dev (FP8)",
      "mflux_name": "dev",
      "pre_quantized": "FP8",
      "pre_quantized_warning": "Model is pre-quantized as FP8. Runtime quantization disabled."
    }
  }
}
```

### **Configuration Options**

| Setting | Description | Options |
|---------|-------------|---------|
| `default_steps` | Default inference steps | Integer (1-50) |
| `default_quantization` | Memory optimization | `"none"`, `"4-bit"`, `"8-bit"` |
| `supports_guidance` | Classifier-free guidance | `true`/`false` |
| `pre_quantized` | Mark pre-quantized models | `"FP8"`, `"4-bit"`, etc. |
| `recommended_t5_encoder` | Suggested T5 for this model | Repository/file path |
| `recommended_clip_encoder` | Suggested CLIP for this model | Repository/file path |

## 🎛 **Node Parameters**

### **Model Selection**
- **Flux Model**: Auto-discovered from cache
- **Quantization**: Dynamic options based on model (pre-quantized models show "none" only)

### **Generation Settings**
- **Width/Height**: Free integer input (512-2048 recommended)
- **Inference Steps**: Free integer input (model-specific recommendations)
- **Guidance Scale**: Float (7.5 for dev, 1.0 for schnell)
- **Seed**: Integer (-1 for random)

### **Text Encoder Settings**
- **T5 Text Encoder**: Dropdown with discovered encoders + "None" option
- **CLIP Text Encoder**: Dropdown with discovered encoders + "None" option

Both encoder parameters support **input connections** for dynamic control.

## 🔮 **Upcoming Features**

- ✅ **Custom T5/CLIP Encoders** (Available now)
- 🚧 **LoRA Support** (Coming soon)
- 🚧 **CUDA Backend** (Future release)
- 🚧 **ControlNet Support** (Planned)
- 🚧 **Img2Img Pipeline** (Planned)

## 💡 **Tips & Best Practices**

### **Memory Optimization**
- Use **4-bit quantization** for lower memory usage
- **Smaller resolutions** (512x512, 768x768) use less memory
- **Fewer steps** reduce memory requirements

### **Quality vs Speed**
- **FLUX.1-dev**: Higher quality, 15-50 steps, supports guidance
- **FLUX.1-schnell**: Faster generation, 1-8 steps, no guidance needed

### **Custom Encoders**
- **"None" option**: Uses model's built-in encoders (recommended for most users)
- **Custom T5**: Try `google/t5-v1_1-large` for different text understanding
- **Custom CLIP**: Experiment with different CLIP variants for style control

## 🐛 **Troubleshooting**

### **"No models found"**
- Download models using `huggingface-cli download <model-name>`
- Check `~/.cache/huggingface/hub/` for model files

### **Memory issues**
- Enable 4-bit or 8-bit quantization
- Reduce image resolution
- Use fewer inference steps

### **MLX not available**
- Ensure you're on Apple Silicon (M1/M2/M3)
- Install MLX: `pip install mlx`

## 📞 **Support**

- **Issues**: Open GitHub issues for bugs/feature requests
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check `node-development-guide-v2.md` for development info

---

**Made with ❤️ for the Griptape Nodes community**
