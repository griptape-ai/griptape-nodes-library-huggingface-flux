import json
import os
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Protocol
from abc import ABC, abstractmethod
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup, ParameterList
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

# Service configuration - no API key needed for local inference
SERVICE = "Flux MLX"

class FluxConfig:
    """Configuration manager for FLUX models"""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "flux_config.json"
        
        with open(config_path) as f:
            self.config = json.load(f)
    
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """Get configuration for a specific model, with global defaults as fallback"""
        global_defaults = self.config["global_defaults"]
        model_config = self.config["models"].get(model_id, {})
        
        # Merge global defaults with model-specific config
        result = global_defaults.copy()
        result.update(model_config)
        
        # If no specific model config exists, infer some settings from model_id
        if model_id not in self.config["models"]:
            # Try to infer mflux_name from model_id
            if "schnell" in model_id.lower():
                result["mflux_name"] = "schnell"
                result["supports_guidance"] = False
                result["default_guidance"] = 1.0
                result["default_steps"] = 4
                result["display_name"] = model_id.split("/")[-1] if "/" in model_id else model_id
            elif "dev" in model_id.lower() or "flux" in model_id.lower():
                result["mflux_name"] = "dev"
                result["supports_guidance"] = True
                result["default_guidance"] = 7.5
                result["default_steps"] = 20
                result["display_name"] = model_id.split("/")[-1] if "/" in model_id else model_id
        
        return result
    
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all supported models"""
        return self.config["models"]
    
    def is_flux_model(self, model_id: str) -> bool:
        """Check if model_id appears to be a FLUX model based on patterns"""
        model_lower = model_id.lower()
        
        # Exclude encoder-only repositories
        encoder_exclusions = [
            "text_encoders" in model_lower,
            "clip_encoders" in model_lower, 
            "t5_encoders" in model_lower,
            model_lower.endswith("/clip"),
            model_lower.endswith("/t5"),
            "encoder" in model_lower and ("clip" in model_lower or "t5" in model_lower)
        ]
        
        if any(encoder_exclusions):
            return False
        
        # Exclude LoRA repositories
        lora_exclusions = [
            "lora" in model_lower,
            "-lora-" in model_lower,
            "lora_" in model_lower,
            model_lower.endswith("_lora"),
            model_lower.endswith("-lora"),
            "adapter" in model_lower and "flux" in model_lower
        ]
        
        if any(lora_exclusions):
            return False
        
        # Check for FLUX patterns
        flux_patterns = [
            "flux" in model_lower,
            "black-forest" in model_lower and "flux" in model_lower,
            "mlx-community" in model_lower and "flux" in model_lower
        ]
        
        return any(flux_patterns)
    
    def extract_mflux_name(self, model_id: str) -> str:
        """Extract mflux model name from model_id, with fallback"""
        model_config = self.get_model_config(model_id)
        return model_config.get("mflux_name", "dev")  # Default to "dev"
    
    def is_model_pre_quantized(self, model_id: str) -> tuple[bool, str, str]:
        """
        Check if model is pre-quantized.
        Returns: (is_pre_quantized, quantization_type, warning_message)
        """
        # First check config file for known models
        model_config = self.get_model_config(model_id)
        if model_config.get("pre_quantized", False):
            quant_type = model_config["pre_quantized"]
            warning = model_config.get("pre_quantized_warning", f"Model is pre-quantized as {quant_type}. Runtime quantization disabled.")
            return True, quant_type, warning
        
        # Aggressive pattern detection for unknown models
        model_lower = model_id.lower()
        
        # Common quantization indicators in model names
        quantization_patterns = {
            "fp8": "FP8",
            "int8": "INT8", 
            "8bit": "8-bit",
            "4bit": "4-bit",
            "int4": "INT4",
            "quantized": "quantized",
            "compressed": "compressed",
            "lite": "lite variant"
        }
        
        for pattern, display_name in quantization_patterns.items():
            if pattern in model_lower:
                warning = f"Model appears to be pre-quantized ({display_name}). Runtime quantization disabled."
                return True, pattern, warning
        
        # No quantization detected
        return False, "", ""
    
    def get_quantization_options(self, model_id: str) -> tuple[list[str], str]:
        """
        Get available quantization options for a model.
        Returns: (options_list, tooltip_message)
        """
        is_pre_quantized, quant_type, warning = self.is_model_pre_quantized(model_id)
        
        if is_pre_quantized:
            return ["none"], warning
        else:
            # Check if we couldn't determine quantization status
            model_config = self.get_model_config(model_id)
            if model_id not in self.config["models"]:
                # Unknown model - show warning but allow options
                warning = "⚠️ Unable to identify if model is pre-quantized. Proceed with caution."
                return self.config["global_defaults"]["quantization_options"], warning
            else:
                # Known model, not pre-quantized
                return self.config["global_defaults"]["quantization_options"], "Model quantization level. Lower bit = faster/less memory, but potentially lower quality."
    
    def get_available_t5_encoders(self) -> list[str]:
        """Scan HuggingFace cache for available T5 encoder files (safetensors only)"""
        try:
            from huggingface_hub import scan_cache_dir
            from pathlib import Path
        except ImportError:
            # If HF not available, return defaults
            return [self.config["global_defaults"]["default_t5_encoder"]]
        
        available_t5 = []
        
        try:
            cache_info = scan_cache_dir()
            
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                
                # Check if this looks like a T5 model or contains T5 encoders
                if self._is_t5_encoder(repo_id):
                    if len(repo.revisions) > 0:
                        # Get the latest revision's snapshot path
                        latest_revision = next(iter(repo.revisions))
                        snapshot_path = Path(latest_revision.snapshot_path)
                        
                        # Look for individual T5 safetensors files
                        for safetensors_file in snapshot_path.glob("*.safetensors"):
                            file_name = safetensors_file.name.lower()
                            # Only include T5 files, not CLIP
                            if "t5" in file_name and "clip" not in file_name:
                                # Format as repo_id/filename for clear identification
                                full_path = f"{repo_id}/{safetensors_file.name}"
                                available_t5.append(full_path)
                                print(f"[FLUX T5] Found T5 encoder file: {full_path}")
                    else:
                        # Fallback to repo-level if no files found
                        available_t5.append(repo_id)
                        print(f"[FLUX T5] Found T5 encoder repo: {repo_id}")
                    
        except Exception as e:
            print(f"[FLUX T5] Error scanning cache: {e}")
            
        # Always include "None" option first for default encoder
        if "None (use model default)" not in available_t5:
            available_t5.insert(0, "None (use model default)")
        
        # Always include default as fallback
        default_t5 = self.config["global_defaults"]["default_t5_encoder"]
        if default_t5 not in available_t5:
            available_t5.insert(1, default_t5)
            
        return available_t5 if available_t5 else ["None (use model default)", default_t5]
    
    def get_recommended_t5_encoder(self, model_id: str) -> str:
        """Get recommended T5 encoder for a specific model"""
        model_config = self.get_model_config(model_id)
        return model_config.get("recommended_t5_encoder", 
                              self.config["global_defaults"]["default_t5_encoder"])
    
    def get_available_clip_encoders(self) -> list[str]:
        """Scan HuggingFace cache for available CLIP encoder files (safetensors only)"""
        try:
            from huggingface_hub import scan_cache_dir
            from pathlib import Path
        except ImportError:
            # If HF not available, return defaults
            return [self.config["global_defaults"]["default_clip_encoder"]]
        
        available_clip = []
        
        try:
            cache_info = scan_cache_dir()
            
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                
                # Check if this looks like a CLIP model or contains CLIP encoders
                if self._is_clip_encoder(repo_id):
                    if len(repo.revisions) > 0:
                        # Get the latest revision's snapshot path
                        latest_revision = next(iter(repo.revisions))
                        snapshot_path = Path(latest_revision.snapshot_path)
                        
                        # Look for individual CLIP safetensors files
                        for safetensors_file in snapshot_path.glob("*.safetensors"):
                            file_name = safetensors_file.name.lower()
                            # Only include CLIP files, not T5
                            if "clip" in file_name and "t5" not in file_name:
                                # Format as repo_id/filename for clear identification
                                full_path = f"{repo_id}/{safetensors_file.name}"
                                available_clip.append(full_path)
                                print(f"[FLUX CLIP] Found CLIP encoder file: {full_path}")
                    else:
                        # Fallback to repo-level if no files found
                        available_clip.append(repo_id)
                        print(f"[FLUX CLIP] Found CLIP encoder repo: {repo_id}")
                    
        except Exception as e:
            print(f"[FLUX CLIP] Error scanning cache: {e}")
            
        # Always include "None" option first for default encoder
        if "None (use model default)" not in available_clip:
            available_clip.insert(0, "None (use model default)")
        
        # Always include default as fallback
        default_clip = self.config["global_defaults"]["default_clip_encoder"]
        if default_clip not in available_clip:
            available_clip.insert(1, default_clip)
            
        return available_clip if available_clip else ["None (use model default)", default_clip]
    
    def get_recommended_clip_encoder(self, model_id: str) -> str:
        """Get recommended CLIP encoder for a specific model"""
        model_config = self.get_model_config(model_id)
        return model_config.get("recommended_clip_encoder", 
                              self.config["global_defaults"]["default_clip_encoder"])
    
    def _is_t5_encoder(self, repo_id: str) -> bool:
        """Check if repo_id appears to be a T5 encoder model or contains T5 encoders"""
        repo_lower = repo_id.lower()
        
        # T5 patterns
        t5_patterns = [
            # Standard T5 models
            "t5" in repo_lower and ("google" in repo_lower or "huggingface" in repo_lower),
            "t5-" in repo_lower,
            "flan-t5" in repo_lower,
            repo_lower.startswith("google/t5"),
            repo_lower.startswith("google/flan-t5"),
            # FLUX-specific T5 encoder repositories  
            "flux_text_encoders" in repo_lower,
            "comfyanonymous" in repo_lower and "flux" in repo_lower,
        ]
        
        return any(t5_patterns)
    
    def _is_clip_encoder(self, repo_id: str) -> bool:
        """Check if repo_id appears to be a CLIP encoder model or contains CLIP encoders"""
        repo_lower = repo_id.lower()
        
        # CLIP patterns
        clip_patterns = [
            # Standard CLIP models
            "clip" in repo_lower and ("openai" in repo_lower or "huggingface" in repo_lower),
            "clip-" in repo_lower,
            repo_lower.startswith("openai/clip"),
            repo_lower.startswith("laion/clip"),
            # FLUX-specific CLIP encoder repositories  
            "flux_text_encoders" in repo_lower,
            "comfyanonymous" in repo_lower and "flux" in repo_lower,
        ]
        
        return any(clip_patterns)

class MLXFluxBackend:
    """MLX-based backend for Apple Silicon using mflux"""
    
    def __init__(self, config: FluxConfig):
        self.config = config
    
    def is_available(self) -> bool:
        try:
            import mlx.core as mx
            import platform
            return platform.machine() == 'arm64' and platform.system() == 'Darwin'
        except ImportError:
            return False
    
    def get_name(self) -> str:
        return "MLX (Apple Silicon)"
    
    def validate_model_id(self, model_id: str) -> bool:
        return self.config.is_flux_model(model_id)
    
    def generate_image(self, **kwargs) -> tuple[Any, dict]:
        try:
            from mflux.flux.flux import Flux1
            from mflux.config.config import Config
            import mlx.core as mx
            import mflux
            
            # Verify we're using our custom fork
            mflux_location = mflux.__file__
            print(f"[FLUX DEBUG] ✅ Successfully imported mflux from: {mflux_location}")
            
            # Check if our custom encoder support exists
            try:
                from mflux.weights.weight_handler import WeightHandler
                has_custom_encoder_method = hasattr(WeightHandler, 'load_custom_encoder_weights')
                print(f"[FLUX DEBUG] Custom encoder support available: {has_custom_encoder_method}")
                
                # Check Flux1.from_name signature to see if it has our encoder parameters
                import inspect
                flux_signature = inspect.signature(Flux1.from_name)
                has_encoder_params = 't5_encoder_path' in flux_signature.parameters
                print(f"[FLUX DEBUG] Flux1.from_name has encoder parameters: {has_encoder_params}")
                print(f"[FLUX DEBUG] Flux1.from_name parameters: {list(flux_signature.parameters.keys())}")
                
            except Exception as e:
                print(f"[FLUX DEBUG] Error checking custom methods: {e}")
            
            print(f"[FLUX DEBUG] ✅ Using griptape-ai mflux fork")
            
        except ImportError as e:
            print(f"[FLUX DEBUG] Import error: {e}")
            import sys
            print(f"[FLUX DEBUG] Python executable: {sys.executable}")
            raise ImportError("mflux not installed. Run: pip install git+https://github.com/griptape-ai/mflux.git@encoder-support")
        
        model_id = kwargs['model_id']
        prompt = kwargs['prompt']  # This is the enhanced prompt for generation
        original_prompts = kwargs.get('original_prompts', [prompt])
        combined_prompt = kwargs.get('combined_prompt', prompt)
        width = kwargs['width']
        height = kwargs['height']
        steps = kwargs['steps']
        guidance = kwargs['guidance']
        seed = kwargs['seed']
        seed_original = kwargs.get('seed_original', seed)
        seed_control = kwargs.get('seed_control', 'fixed')
        quantization = kwargs.get('quantization', 4)
        t5_encoder = kwargs.get('t5_encoder', self.config.config["global_defaults"]["default_t5_encoder"])
        clip_encoder = kwargs.get('clip_encoder', self.config.config["global_defaults"]["default_clip_encoder"])
        loras = kwargs.get('loras', [])
        
        # Get model config (with global defaults if needed)
        model_config = self.config.get_model_config(model_id)
        mflux_model = model_config['mflux_name']
        
        # Handle "None" option for default encoders
        use_custom_t5 = t5_encoder and not t5_encoder.startswith("None")
        use_custom_clip = clip_encoder and not clip_encoder.startswith("None")
        t5_encoder_path = t5_encoder if use_custom_t5 else None
        clip_encoder_path = clip_encoder if use_custom_clip else None
        
        # Process LoRA list to extract paths and scales
        lora_paths = []
        lora_scales = []
        
        if loras:
            print(f"[FLUX LoRA] Processing {len(loras)} LoRA(s)")
            for i, lora in enumerate(loras):
                if isinstance(lora, dict) and 'path' in lora and 'scale' in lora:
                    lora_path = lora['path']
                    lora_scale = float(lora['scale'])
                    lora_paths.append(lora_path)
                    lora_scales.append(lora_scale)
                    
                    # Log LoRA info
                    lora_name = lora.get('name', lora_path.split('/')[-1] if '/' in lora_path else lora_path)
                    print(f"[FLUX LoRA] {i+1}. {lora_name} (scale: {lora_scale})")
                    
                    # Check for gated models
                    if lora.get('gated', False):
                        print(f"[FLUX LoRA] Warning: {lora_name} is gated - ensure you have access")
                else:
                    print(f"[FLUX LoRA] Warning: Skipping invalid LoRA object: {lora}")
        else:
            print(f"[FLUX LoRA] No LoRAs provided")
        
        # Convert to None if empty (mflux expects None for no LoRAs)
        lora_paths = lora_paths if lora_paths else None
        lora_scales = lora_scales if lora_scales else None
        
        # Debug logging for encoder selection  
        print(f"[FLUX T5] Selected T5 encoder: {t5_encoder}")
        if use_custom_t5:
            print(f"[FLUX T5] Using custom T5 encoder via modified mflux fork")
            print(f"[FLUX T5] Custom encoder will be loaded and applied to the model")
        else:
            print(f"[FLUX T5] Using model's default T5 encoder")
        
        print(f"[FLUX CLIP] Selected CLIP encoder: {clip_encoder}")
        if use_custom_clip:
            print(f"[FLUX CLIP] Using custom CLIP encoder via modified mflux fork")
            print(f"[FLUX CLIP] Custom encoder will be loaded and applied to the model")
        else:
            print(f"[FLUX CLIP] Using model's default CLIP encoder")
        
        # Memory management for MLX
        import mlx.core as mx
        print(f"[FLUX DEBUG] Memory before model creation: {mx.get_peak_memory() / 1e9:.2f} GB peak")
        
        # Create Flux1 model with quantization and custom encoders
        quantize_param = None if quantization == "none" else int(quantization.replace("-bit", ""))
        print(f"[FLUX DEBUG] Creating FLUX model with quantization: {quantize_param}")
        
        flux_model = Flux1.from_name(
            model_name=mflux_model, 
            quantize=quantize_param,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            t5_encoder_path=t5_encoder_path,
            clip_encoder_path=clip_encoder_path
        )
        
        print(f"[FLUX DEBUG] Memory after model creation: {mx.get_peak_memory() / 1e9:.2f} GB peak")
        print(f"[FLUX DEBUG] Model created successfully, preparing generation config")
        
        # Create generation config
        config = Config(
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance=guidance if model_config['supports_guidance'] else 1.0
        )
        
        # Generate image
        print(f"[FLUX DEBUG] Memory before generation: {mx.get_peak_memory() / 1e9:.2f} GB peak")
        print(f"[FLUX DEBUG] Starting generation with {steps} steps...")
        
        result = flux_model.generate_image(
            seed=seed,
            prompt=prompt,
            config=config
        )
        
        print(f"[FLUX DEBUG] Memory after generation: {mx.get_peak_memory() / 1e9:.2f} GB peak")
        print(f"[FLUX DEBUG] Generation completed successfully")
        
        # Extract PIL image from result
        image = result.image
        
        generation_info = {
            "backend": "MLX",
            "model": model_id,
            "mflux_model": mflux_model,
            "original_prompts": original_prompts,
            "prompt_count": len(original_prompts),
            "combined_prompt": combined_prompt,
            "enhanced_prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance,
            "seed_original": seed_original,
            "seed_control": seed_control,
            "seed_used": seed,
            "quantization": quantization,
            "t5_encoder": t5_encoder,
            "clip_encoder": clip_encoder,
            "loras_used": loras,
            "lora_count": len(loras) if loras else 0
        }
        
        return image, generation_info

class FluxInference(ControlNode):
    """FLUX inference node optimized for Apple Silicon using MLX"""
    
    # Class variable to track last used seed across instances (ComfyUI-style)
    _last_used_seed: int = 12345
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Flux MLX"
        
        # Load configuration
        self.flux_config = FluxConfig()
        
        # Initialize MLX backend
        self._backend = MLXFluxBackend(self.flux_config)
        if not self._backend.is_available():
            raise RuntimeError("MLX backend not available. This library requires Apple Silicon (M1/M2/M3) and 'mlx' + 'mflux' packages.")
        
        backend_name = self._backend.get_name()
        self.description = f"FLUX inference optimized for Apple Silicon using {backend_name}. Supports FLUX.1-dev and FLUX.1-schnell with native MLX acceleration."

        # Initialize available models by scanning cache
        available_models = self._scan_available_models()
        
        # Model Selection Group - Always visible
        with ParameterGroup(name=f"Model Selection ({backend_name})") as model_group:
            self.add_parameter(
                Parameter(
                    name="model",
                    tooltip="Flux model to use for generation. Models are auto-discovered from HuggingFace cache.",
                    type="str",
                    default_value=available_models[0] if available_models else "",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=available_models)},
                    ui_options={"display_name": "Flux Model"}
                )
            )
            
            # Add quantization dropdown - initial setup
            default_model = available_models[0] if available_models else ""
            initial_options, initial_tooltip = self.flux_config.get_quantization_options(default_model)
            default_quantization = "none" if len(initial_options) == 1 else "4-bit"
            
            self.add_parameter(
                Parameter(
                    name="quantization",
                    tooltip=initial_tooltip,
                    type="str",
                    default_value=default_quantization,
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=initial_options)},
                    ui_options={"display_name": "Quantization"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="main_prompt",
                    tooltip="Primary text description for image generation",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    default_value="",
                    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                    ui_options={
                        "display_name": "Main Prompt",
                        "multiline": True,
                        "placeholder_text": "Describe the image you want to generate..."
                    }
                )
            )
            
            self.add_parameter(
                ParameterList(
                    name="additional_prompts",
                    input_types=["str", "list[str]"],
                    default_value=[],
                    tooltip="Optional additional prompts to combine with main prompt.\nConnect multiple prompt sources or manually add prompts.",
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": "Additional Prompts"}
                )
            )

        # Shared Generation Settings - Always visible  
        with ParameterGroup(name="Generation Settings") as gen_group:
            global_defaults = self.flux_config.config["global_defaults"]
            
            self.add_parameter(
                Parameter(
                    name="width",
                    tooltip="Width of generated image in pixels. Recommended: 512-1536. Higher resolutions require more memory.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    default_value=global_defaults["default_width"],
                    allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT, ParameterMode.OUTPUT},
                    ui_options={"display_name": "Width"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="height", 
                    tooltip="Height of generated image in pixels. Recommended: 512-1536. Higher resolutions require more memory.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    default_value=global_defaults["default_height"],
                    allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT, ParameterMode.OUTPUT},
                    ui_options={"display_name": "Height"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="guidance_scale",
                    tooltip="How closely to follow the prompt (higher = more adherence)",
                    type="float",
                    default_value=global_defaults["default_guidance"],
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Guidance Scale"}
                )
            )
            
            # Initialize steps parameter with default slider range (will be updated based on model)
            default_max_steps = global_defaults.get("max_steps", 50)
            
            self.add_parameter(
                Parameter(
                    name="steps",
                    tooltip="Number of inference steps. More steps = higher quality but slower generation. FLUX.1-dev: 15-50 steps, FLUX.1-schnell: 1-8 steps recommended.",
                    type="int",
                    default_value=global_defaults["default_steps"],
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Slider(min_val=1, max_val=default_max_steps)},
                    ui_options={"display_name": "Inference Steps"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="seed",
                    tooltip="Seed value for reproducible generation",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    default_value=12345,
                    allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT, ParameterMode.OUTPUT},
                    ui_options={"display_name": "Seed"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="seed_control",
                    tooltip="Seed control mode: Fixed (use exact value), Increment (+1 each run), Decrement (-1 each run), Randomize (new random each run)",
                    type="str",
                    default_value="randomize",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=["fixed", "increment", "decrement", "randomize"])},
                    ui_options={"display_name": "Control"}
                )
            )
        
        gen_group.ui_options = {"collapsed": False}
        self.add_node_element(gen_group)

        # Text Encoder Configuration - Always visible
        with ParameterGroup(name="Text Encoder Settings") as encoder_group:
            # Get available T5 encoders
            available_t5_encoders = self.flux_config.get_available_t5_encoders()
            # Default to "None" to use model's built-in encoder
            default_t5 = "None (use model default)"
            
            self.add_parameter(
                Parameter(
                    name="t5_encoder",
                    tooltip="T5 text encoder model selection. 'None' uses the model's built-in encoder. Custom encoders require safetensors format.",
                    type="str",
                    input_types=["str"],
                    default_value=default_t5,
                    allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                    traits={Options(choices=available_t5_encoders)},
                    ui_options={"display_name": "T5 Text Encoder"}
                )
            )
            
            # Get available CLIP encoders
            available_clip_encoders = self.flux_config.get_available_clip_encoders()
            # Default to "None" to use model's built-in encoder
            default_clip = "None (use model default)"
            
            self.add_parameter(
                Parameter(
                    name="clip_encoder",
                    tooltip="CLIP text encoder model selection. 'None' uses the model's built-in encoder. Custom encoders require safetensors format.",
                    type="str",
                    input_types=["str"],
                    default_value=default_clip,
                    allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                    traits={Options(choices=available_clip_encoders)},
                    ui_options={"display_name": "CLIP Text Encoder"}
                )
            )

        encoder_group.ui_options = {"collapsed": False}
        self.add_node_element(encoder_group)

        # LoRA Settings - Always visible
        with ParameterGroup(name="LoRA Settings") as lora_group:
            self.add_parameter(
                ParameterList(
                    name="loras",
                    input_types=["dict", "list[dict]"],
                    default_value=[],
                    tooltip="Connect LoRA objects from HuggingFace LoRA Discovery nodes.\nEach LoRA dict should contain 'path' and 'scale' keys.\nMultiple LoRAs will be applied in sequence.",
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": "LoRA Models"}
                )
            )

        lora_group.ui_options = {"collapsed": False}
        self.add_node_element(lora_group)

        # Set initial quantization parameter options based on default model
        if available_models:
            self._update_quantization_options(available_models[0])
            self._update_steps_range(available_models[0])

        # Output parameters
        self.add_parameter(
            Parameter(
                name="image",
                output_type="ImageUrlArtifact",
                tooltip="Generated image",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Generated Image"}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="generation_info",
                output_type="str",
                tooltip="Generation metadata and parameters used",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Generation Info", "multiline": True}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="status",
                output_type="str", 
                tooltip="Real-time generation status and progress",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Status", "multiline": True}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="actual_seed",
                output_type="int",
                tooltip="The actual seed value used for generation (after applying seed control mode)",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Actual Seed Used"}
            )
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Callback triggered when parameter values change"""        
        if parameter.name == "model":
            # Update quantization parameter options when model changes
            self._update_quantization_options(value)
            # Update steps range when model changes
            self._update_steps_range(value)
            # Update T5 encoder recommendation when model changes
            self._update_t5_encoder_recommendation(value)
            # Update CLIP encoder recommendation when model changes
            self._update_clip_encoder_recommendation(value)
    
    def after_incoming_connection(self, source_node, source_parameter: Parameter, target_parameter: Parameter) -> None:
        """Handle when connections to ParameterList are made"""
        print(f"[FLUX DEBUG] Connection added: {source_node.name} -> {target_parameter.name}")
        if target_parameter.name == "loras":
            print(f"[FLUX DEBUG] LoRA connection added from {source_node.name}")
        elif target_parameter.name == "additional_prompts":
            print(f"[FLUX DEBUG] Additional prompt connection added from {source_node.name}")
    

    
    def _cleanup_parameter_list(self, param_name: str, source_node_name: str) -> None:
        """Helper to clean up empty elements from a ParameterList"""
        print(f"[FLUX DEBUG] _cleanup_parameter_list called with param_name: '{param_name}'")
        
        # Extract base parameter name (handle unique ID suffixes)
        base_param_name = param_name
        if "_ParameterListUniqueParamID_" in param_name:
            base_param_name = param_name.split("_ParameterListUniqueParamID_")[0]
            print(f"[FLUX DEBUG] Extracted base parameter name: '{base_param_name}'")
        
        if base_param_name == "loras":
            print(f"[FLUX DEBUG] LoRA connection removed from {source_node_name}")
            try:
                current_loras = self.get_parameter_list_value("loras")
                print(f"[FLUX DEBUG] LoRA list before cleanup: {len(current_loras) if current_loras else 0} items")
                print(f"[FLUX DEBUG] LoRA list contents: {current_loras}")
                
                if current_loras:
                    cleaned_loras = [lora for lora in current_loras if lora is not None and lora != {} and lora != ""]
                    print(f"[FLUX DEBUG] LoRA list after cleanup: {len(cleaned_loras)} items")
                    print(f"[FLUX DEBUG] Cleaned LoRA contents: {cleaned_loras}")
                    
                    if len(cleaned_loras) != len(current_loras):
                        print(f"[FLUX DEBUG] Cleaning up {len(current_loras) - len(cleaned_loras)} empty LoRA element(s)")
                        self._reset_parameter_list("loras", cleaned_loras)
                    else:
                        print(f"[FLUX DEBUG] No empty LoRA elements to clean up")
                else:
                    print(f"[FLUX DEBUG] LoRA list is empty, forcing UI refresh to remove empty elements")
                    # Force UI refresh even when empty to remove lingering empty elements
                    self._reset_parameter_list("loras", [])
                        
            except Exception as e:
                print(f"[FLUX DEBUG] Error cleaning LoRA list after removal: {e}")
                import traceback
                traceback.print_exc()
                
        elif base_param_name == "additional_prompts":
            print(f"[FLUX DEBUG] Additional prompt connection removed from {source_node_name}")
            try:
                current_prompts = self.get_parameter_list_value("additional_prompts")
                print(f"[FLUX DEBUG] Additional prompts before cleanup: {len(current_prompts) if current_prompts else 0} items")
                print(f"[FLUX DEBUG] Additional prompts contents: {current_prompts}")
                
                if current_prompts:
                    cleaned_prompts = [p for p in current_prompts if p is not None and p != "" and str(p).strip() != ""]
                    print(f"[FLUX DEBUG] Additional prompts after cleanup: {len(cleaned_prompts)} items")
                    print(f"[FLUX DEBUG] Cleaned prompts contents: {cleaned_prompts}")
                    
                    if len(cleaned_prompts) != len(current_prompts):
                        print(f"[FLUX DEBUG] Cleaning up {len(current_prompts) - len(cleaned_prompts)} empty prompt element(s)")
                        self._reset_parameter_list("additional_prompts", cleaned_prompts)
                    else:
                        print(f"[FLUX DEBUG] No empty prompt elements to clean up")
                else:
                    print(f"[FLUX DEBUG] Additional prompts list is empty, forcing UI refresh to remove empty elements")
                    # Force UI refresh even when empty to remove lingering empty elements
                    self._reset_parameter_list("additional_prompts", [])
                        
            except Exception as e:
                print(f"[FLUX DEBUG] Error cleaning additional prompts after removal: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[FLUX DEBUG] Unknown parameter for cleanup: '{base_param_name}' (original: '{param_name}')")
    
    def _reset_parameter_list(self, param_name: str, new_values: list) -> None:
        """Reset a ParameterList by clearing and repopulating it"""
        print(f"[FLUX DEBUG] _reset_parameter_list called for '{param_name}' with {len(new_values)} values")
        
        try:
            # Try to set the parameter value directly (this might work for ParameterList)
            self.set_parameter_value(param_name, new_values)
            print(f"[FLUX DEBUG] Successfully reset {param_name} to {len(new_values)} items via direct set")
        except Exception as direct_error:
            print(f"[FLUX DEBUG] Direct reset failed for {param_name}: {direct_error}")
            
            # Alternative approach: try to rebuild by appending values
            try:
                # Clear by setting to empty list first
                self.set_parameter_value(param_name, [])
                print(f"[FLUX DEBUG] Cleared {param_name} to empty list")
                
                # Then append each valid value
                for i, value in enumerate(new_values):
                    self.append_value_to_parameter(param_name, value)
                    print(f"[FLUX DEBUG] Appended value {i+1}: {value}")
                    
                print(f"[FLUX DEBUG] Successfully rebuilt {param_name} with {len(new_values)} items via append")
                
            except Exception as rebuild_error:
                print(f"[FLUX DEBUG] Rebuild failed for {param_name}: {rebuild_error}")
                
                # Last resort: try to force parameter refresh
                try:
                    param = self.get_parameter_by_name(param_name)
                    if param:
                        print(f"[FLUX DEBUG] Attempting parameter refresh for {param_name}")
                        # Force parameter to notify change
                        param.default_value = []
                        param.value = []
                        print(f"[FLUX DEBUG] Forced parameter values to empty for {param_name}")
                    else:
                        print(f"[FLUX DEBUG] Could not find parameter {param_name}")
                        
                except Exception as refresh_error:
                    print(f"[FLUX DEBUG] Parameter refresh failed: {refresh_error}")
                    print(f"[FLUX DEBUG] ParameterList cleanup may require manual user action")
    
    def on_griptape_event(self, event) -> None:
        """Handle general Griptape events - might catch node deletions"""
        print(f"[FLUX DEBUG] Griptape event: {type(event).__name__}")
        
        # Check if this is a connection or node-related event
        if hasattr(event, 'node') or hasattr(event, 'connection'):
            print(f"[FLUX DEBUG] Event details: {event}")
    
    # Try alternative callback signatures in case the current one is wrong
    def after_incoming_connection_removed(self, source_node, source_parameter: Parameter, target_parameter: Parameter) -> None:
        """Handle cleanup when connections to ParameterList are removed"""
        print(f"[FLUX DEBUG] *** CONNECTION REMOVED: {source_node.name} -> {target_parameter.name} ***")
        self._cleanup_parameter_list(target_parameter.name, source_node.name)
        
    def after_connection_removed(self, **kwargs) -> None:
        """Catch-all for connection removal with any signature"""
        print(f"[FLUX DEBUG] *** after_connection_removed called with: {kwargs} ***")
    
    def _update_option_choices(self, param: str, choices: list[str], default: str = None) -> None:
        """Update parameter option choices and optionally set a new default value."""
        parameter = self.get_parameter_by_name(param)
        if parameter is not None:
            trait = parameter.find_element_by_id("Options")
            if trait and isinstance(trait, Options):
                print(f"[FLUX] Before update - trait.choices: {trait.choices}")
                trait.choices = choices
                print(f"[FLUX] After update - trait.choices: {trait.choices}")

                if default and default in choices:
                    parameter.default_value = default
                    self.set_parameter_value(param, default)
                    print(f"[FLUX] Set parameter value to: {default}")
            else:
                print(f"[FLUX] Options trait not found for parameter '{param}'")
        else:
            print(f"[FLUX] Parameter '{param}' not found for updating option choices.")

    def _update_quantization_options(self, model_id: str) -> None:
        """Update quantization parameter options based on model selection"""
        if not model_id:
            return
            
        # Get new options and tooltip for this model
        new_options, new_tooltip = self.flux_config.get_quantization_options(model_id)
        
        # Determine appropriate default
        if len(new_options) == 1:
            new_default = "none"  # Pre-quantized models
        else:
            new_default = "4-bit" if "4-bit" in new_options else new_options[0]  # Regular models
        
        print(f"[FLUX] Updating quantization options: {new_options}, default: {new_default}")
        
        # Update options like the OpenAI example
        self._update_option_choices(param="quantization", choices=new_options, default=new_default)
        
        # Update tooltip
        quant_param = self.get_parameter_by_name("quantization")
        if quant_param:
            quant_param.tooltip = new_tooltip
    
    def _update_steps_range(self, model_id: str) -> None:
        """Update steps parameter default and tooltip based on model selection"""
        if not model_id:
            return
            
        # Get model config to determine optimal step ranges
        model_config = self.flux_config.get_model_config(model_id)
        mflux_name = model_config.get("mflux_name", "dev")
        
        steps_param = self.get_parameter_by_name("steps")
        if not steps_param:
            return
            
        if mflux_name == "schnell":
            # FLUX.1-schnell: optimized for 1-8 steps
            default_steps = 4
            max_steps = model_config.get("max_steps", 8)
            tooltip = f"Number of inference steps. FLUX.1-schnell is optimized for 1-{max_steps} steps (fast generation)."
        else:
            # FLUX.1-dev: works best with 15-50 steps  
            default_steps = 15
            max_steps = model_config.get("max_steps", 50)
            tooltip = f"Number of inference steps. FLUX.1-dev works best with 15-{max_steps} steps (higher quality)."
        
        print(f"[FLUX] Updating steps for {model_id} ({mflux_name}): default={default_steps}, max={max_steps}")
        
        # Update slider min/max values
        slider_trait = steps_param.find_element_by_id("Slider")
        if slider_trait and isinstance(slider_trait, Slider):
            slider_trait.min_val = 1
            slider_trait.max_val = max_steps
            print(f"[FLUX] Updated steps slider range: 1-{max_steps}")
        
        # Update default value and tooltip
        steps_param.default_value = default_steps
        steps_param.tooltip = tooltip
        self.set_parameter_value("steps", default_steps)
    
    def _update_t5_encoder_recommendation(self, model_id: str) -> None:
        """Update T5 encoder choices without changing user selection"""
        if not model_id:
            return
            
        # Get recommended T5 encoder for this model
        recommended_t5 = self.flux_config.get_recommended_t5_encoder(model_id)
        
        # Get available encoders
        available_t5_encoders = self.flux_config.get_available_t5_encoders()
        
        # Update T5 encoder choices but preserve user's current selection
        self._update_option_choices(param="t5_encoder", choices=available_t5_encoders)  # No default!
        
        print(f"[FLUX T5] Updated T5 encoder choices for {model_id}, recommended: {recommended_t5}")
    
    def _update_clip_encoder_recommendation(self, model_id: str) -> None:
        """Update CLIP encoder choices without changing user selection"""
        if not model_id:
            return
            
        # Get recommended CLIP encoder for this model
        recommended_clip = self.flux_config.get_recommended_clip_encoder(model_id)
        
        # Get available encoders
        available_clip_encoders = self.flux_config.get_available_clip_encoders()
        
        # Update CLIP encoder choices but preserve user's current selection
        self._update_option_choices(param="clip_encoder", choices=available_clip_encoders)  # No default!
        
        print(f"[FLUX CLIP] Updated CLIP encoder choices for {model_id}, recommended: {recommended_clip}")

    def _scan_available_models(self) -> list[str]:
        """Dynamically scan HuggingFace cache for FLUX models by analyzing model structure."""
        try:
            from huggingface_hub import scan_cache_dir
            import json
            from pathlib import Path
        except ImportError:
            # If HF not available, return config models as fallback
            return list(self.flux_config.get_supported_models().keys())
        
        available = []
        try:
            cache_info = scan_cache_dir()
            
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                
                # First check if it has FLUX in the name at all
                if not self.flux_config.is_flux_model(repo_id):
                    continue
                
                # Now analyze the actual model structure
                if len(repo.revisions) > 0:
                    # Get the latest revision's snapshot path
                    latest_revision = next(iter(repo.revisions))
                    snapshot_path = Path(latest_revision.snapshot_path)
                    
                    if self._is_base_flux_model(snapshot_path, repo_id):
                        available.append(repo_id)
                        print(f"[FLUX SCAN] Found base FLUX model: {repo_id}")
                    else:
                        print(f"[FLUX SCAN] Skipping specialized FLUX model: {repo_id}")
                        
        except Exception as e:
            print(f"[FLUX SCAN] Error scanning cache: {e}")
            # If scanning fails, return config models as fallback
            available = list(self.flux_config.get_supported_models().keys())
            
        # Always return at least one option (fallback to config models)
        if not available:
            available = list(self.flux_config.get_supported_models().keys())
            print("[FLUX SCAN] No models found in cache, using config defaults")
        
        print(f"[FLUX SCAN] Available models: {available}")
        return available
    
    def _is_base_flux_model(self, snapshot_path: Path, repo_id: str) -> bool:
        """Analyze model structure to determine if it's a base FLUX text-to-image model."""
        try:
            # First check: exclude encoder-only repositories by structure
            if self._is_encoder_only_repository(snapshot_path, repo_id):
                print(f"[FLUX SCAN] {repo_id}: Encoder-only repository detected, skipping")
                return False
            
            # Check for model_index.json (indicates diffusers pipeline)
            model_index_path = snapshot_path / "model_index.json"
            if model_index_path.exists():
                with open(model_index_path) as f:
                    model_index = json.load(f)
                
                # Check if it's a FLUX pipeline with the right components
                if model_index.get("_class_name") in ["FluxPipeline", "FlowMatchEulerDiscreteScheduler"]:
                    # Make sure it has the core components for text-to-image
                    required_components = ["transformer", "scheduler", "vae"]
                    has_required = all(comp in model_index for comp in required_components)
                    
                    # Make sure it's NOT a ControlNet (they have controlnet in components)
                    is_controlnet = "controlnet" in model_index
                    
                    if has_required and not is_controlnet:
                        print(f"[FLUX SCAN] {repo_id}: Valid FLUX pipeline")
                        return True
                    elif is_controlnet:
                        print(f"[FLUX SCAN] {repo_id}: ControlNet detected, skipping")
                        return False
            
            # Check individual model config files
            config_path = snapshot_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                
                # Check if it's a ControlNet by architecture
                if config.get("_class_name") == "FluxControlNetModel":
                    print(f"[FLUX SCAN] {repo_id}: ControlNet config detected, skipping")
                    return False
                
                # Check if it's a transformer model (the main FLUX component)
                if config.get("_class_name") == "FluxTransformer2DModel":
                    print(f"[FLUX SCAN] {repo_id}: FLUX transformer detected")
                    return True
            
            # Fallback: check for common ControlNet indicators in the directory structure
            controlnet_indicators = [
                "controlnet",
                "control_net", 
                "canny",
                "depth",
                "pose",
                "inpaint",
                "fill"
            ]
            
            repo_lower = repo_id.lower()
            if any(indicator in repo_lower for indicator in controlnet_indicators):
                print(f"[FLUX SCAN] {repo_id}: ControlNet/specialized model detected by name patterns")
                return False
            
            # If we can't determine definitively, default to including it
            print(f"[FLUX SCAN] {repo_id}: Assuming base model (no clear specialization detected)")
            return True
            
        except Exception as e:
            print(f"[FLUX SCAN] Error analyzing {repo_id}: {e}")
            # When in doubt, include it
            return True
    
    def _is_encoder_only_repository(self, snapshot_path: Path, repo_id: str) -> bool:
        """Check if this repository only contains text encoders (T5/CLIP)"""
        try:
            # List all files in the repository
            all_files = list(snapshot_path.rglob("*.safetensors"))
            
            # Check if it only contains encoder files and no core FLUX components
            has_transformer = any("transformer" in str(f) for f in all_files)
            has_vae = any("vae" in str(f) for f in all_files)
            has_scheduler = (snapshot_path / "scheduler").exists()
            
            # If it has no core FLUX components, it's likely an encoder-only repository
            if not (has_transformer or has_vae or has_scheduler):
                # Double-check by looking for encoder patterns
                encoder_files = [f for f in all_files if any(pattern in str(f).lower() for pattern in ["clip", "t5", "text_encoder"])]
                if encoder_files and len(encoder_files) == len(all_files):
                    return True
            
            return False
            
        except Exception as e:
            print(f"[FLUX SCAN] Error checking encoder-only status for {repo_id}: {e}")
            return False

    def process(self) -> AsyncResult:
        """Generate image using MLX backend."""
        
        def generate_image() -> str:
            try:
                # Get parameters
                model_id = self.get_parameter_value("model")
                main_prompt = self.get_parameter_value("main_prompt")
                additional_prompts = self.get_parameter_list_value("additional_prompts")
                width = int(self.get_parameter_value("width"))
                height = int(self.get_parameter_value("height"))
                steps = int(self.get_parameter_value("steps"))
                guidance_scale = float(self.get_parameter_value("guidance_scale"))
                seed_value = int(self.get_parameter_value("seed"))
                seed_control = self.get_parameter_value("seed_control")
                quantization = self.get_parameter_value("quantization")
                t5_encoder = self.get_parameter_value("t5_encoder")
                clip_encoder = self.get_parameter_value("clip_encoder")
                loras = self.get_parameter_list_value("loras")
                
                # Validate inputs
                if not model_id:
                    raise ValueError("No Flux model selected. Please select a model.")
                
                # Validate main prompt
                if not main_prompt or not main_prompt.strip():
                    raise ValueError("Please enter a main prompt to describe the image you want to generate.")
                
                # Combine main prompt with additional prompts
                all_prompts = [main_prompt.strip()]
                if additional_prompts:
                    # Filter out empty additional prompts
                    valid_additional = [p.strip() for p in additional_prompts if p and p.strip()]
                    all_prompts.extend(valid_additional)
                
                # Limit to 5 total prompts
                valid_prompts = all_prompts[:5]
                
                # Debug logging for prompt filtering
                total_prompts_provided = 1 + (len(additional_prompts) if additional_prompts else 0)
                filtered_count = total_prompts_provided - len(valid_prompts)
                if filtered_count > 0:
                    print(f"[FLUX PROMPTS] Filtered out {filtered_count} empty additional prompt(s)")
                
                # Combine prompts with intelligent separators
                if len(valid_prompts) == 1:
                    combined_prompt = valid_prompts[0]
                else:
                    # For multiple prompts, combine with commas and clean up
                    combined_prompt = ", ".join(valid_prompts)
                    # Clean up multiple commas/spaces
                    import re
                    combined_prompt = re.sub(r',\s*,', ',', combined_prompt)  # Remove duplicate commas
                    combined_prompt = re.sub(r'\s+', ' ', combined_prompt)      # Normalize whitespace
                    combined_prompt = combined_prompt.strip(', ')              # Remove leading/trailing separators
                
                print(f"[FLUX PROMPTS] Combined {len(valid_prompts)} prompt(s): '{combined_prompt[:100]}{'...' if len(combined_prompt) > 100 else ''}'")
                prompt = combined_prompt
                
                # Validate model compatibility with backend
                if not self._backend.validate_model_id(model_id):
                    raise ValueError(f"Model {model_id} not supported by {self._backend.get_name()} backend")
                
                # Validate parameters
                model_config = self.flux_config.get_model_config(model_id)
                max_steps = model_config.get('max_steps', 50)
                
                # Validate steps
                if steps < 1:
                    raise ValueError("Steps must be at least 1.")
                if steps > max_steps:
                    self.publish_update_to_parameter("status", f"⚠️ Warning: {steps} steps exceeds recommended maximum of {max_steps} for this model")
                    print(f"[FLUX DEBUG] High step count warning: {steps} > {max_steps}")
                
                # Validate resolution
                if width < 64 or height < 64:
                    raise ValueError("Width and height must be at least 64 pixels.")
                if width > 2048 or height > 2048:
                    self.publish_update_to_parameter("status", f"⚠️ Warning: {width}x{height} is very high resolution and may cause memory issues")
                    print(f"[FLUX DEBUG] High resolution warning: {width}x{height}")
                
                # Check if dimensions are multiples of 8 (optimal for diffusion models)
                if width % 8 != 0 or height % 8 != 0:
                    self.publish_update_to_parameter("status", f"💡 Tip: Dimensions divisible by 8 work best. Current: {width}x{height}")
                
                # Warn about memory usage
                total_pixels = width * height
                if steps > 30:
                    self.publish_update_to_parameter("status", f"⚠️ High step count ({steps}) may cause memory issues")
                if total_pixels > 1024 * 1024:
                    self.publish_update_to_parameter("status", f"⚠️ High resolution ({width}x{height}) requires substantial memory")
                
                # Validate quantization settings for pre-quantized models
                is_pre_quantized, quant_type, warning = self.flux_config.is_model_pre_quantized(model_id)
                if is_pre_quantized and quantization != "none":
                    self.publish_update_to_parameter("status", f"⚠️ Warning: Model is pre-quantized ({quant_type}), forcing quantization to 'none'")
                    print(f"[FLUX DEBUG] Pre-quantized model detected, overriding quantization: {quantization} → none")
                    quantization = "none"
                
                # Handle seed control (ComfyUI-style)
                if seed_control == "fixed":
                    actual_seed = seed_value
                elif seed_control == "increment":
                    actual_seed = FluxInference._last_used_seed + 1
                elif seed_control == "decrement":
                    actual_seed = FluxInference._last_used_seed - 1
                elif seed_control == "randomize":
                    import random
                    actual_seed = random.randint(0, 2**32 - 1)
                else:
                    actual_seed = seed_value  # fallback
                
                # Ensure seed is in valid range
                actual_seed = max(0, min(actual_seed, 2**32 - 1))
                
                # Update last used seed for next run
                FluxInference._last_used_seed = actual_seed
                
                # Log seed control info
                print(f"[FLUX SEED] Control: {seed_control}, Original: {seed_value}, Used: {actual_seed}")
                
                # Enhance simple prompts for better generation
                enhanced_prompt = prompt.strip()
                if len(enhanced_prompt.split()) < 3:
                    if enhanced_prompt.lower() == "capybara":
                        enhanced_prompt = "a cute capybara sitting on grass, detailed, high quality, photorealistic"
                        enhance_msg = f"📝 Enhanced simple prompt: '{prompt.strip()}' → '{enhanced_prompt}'"
                        self.publish_update_to_parameter("status", enhance_msg)
                        print(f"[FLUX DEBUG] {enhance_msg}")
                
                # Use user's steps parameter (model_config already retrieved for validation)
                num_steps = steps
                
                # Override guidance for schnell model
                if not model_config['supports_guidance']:
                    guidance_scale = 1.0
                
                self.publish_update_to_parameter("status", f"🚀 Generating with {self._backend.get_name()} backend...")
                if len(valid_prompts) > 1:
                    self.publish_update_to_parameter("status", f"📝 Combined {len(valid_prompts)} prompts: '{enhanced_prompt[:50]}{'...' if len(enhanced_prompt) > 50 else ''}'")
                else:
                    self.publish_update_to_parameter("status", f"📝 Prompt: '{enhanced_prompt[:50]}{'...' if len(enhanced_prompt) > 50 else ''}'")
                self.publish_update_to_parameter("status", f"⚙️ Settings: {width}x{height}, {steps} steps, guidance={guidance_scale}, quantization={quantization}")
                
                # Show seed control info
                if seed_control == "increment":
                    self.publish_update_to_parameter("status", f"🔢 Seed: {actual_seed} (incremented from {FluxInference._last_used_seed - 1})")
                elif seed_control == "decrement":
                    self.publish_update_to_parameter("status", f"🔢 Seed: {actual_seed} (decremented from {FluxInference._last_used_seed + 1})")
                elif seed_control == "randomize":
                    self.publish_update_to_parameter("status", f"🔢 Seed: {actual_seed} (randomized)")
                else:
                    self.publish_update_to_parameter("status", f"🔢 Seed: {actual_seed} (fixed)")
                
                # Memory optimization hint
                if quantization == "none" and steps > 20:
                    self.publish_update_to_parameter("status", f"💡 Tip: For better memory usage, try 4-bit quantization or fewer steps")
                elif quantization in ["4-bit", "8-bit"]:
                    self.publish_update_to_parameter("status", f"✅ Using {quantization} quantization for memory efficiency")
                print(f"[FLUX DEBUG] Backend: {self._backend.get_name()}")
                print(f"[FLUX DEBUG] Model: {model_id}")
                print(f"[FLUX DEBUG] Original prompts: {valid_prompts}")
                print(f"[FLUX DEBUG] Combined prompt: '{prompt}'")
                print(f"[FLUX DEBUG] Enhanced prompt: '{enhanced_prompt}'")
                print(f"[FLUX DEBUG] Quantization: {quantization}")
                print(f"[FLUX DEBUG] T5 Encoder: {t5_encoder}")
                print(f"[FLUX DEBUG] Model config: {model_config}")
                
                # Generate using backend
                generated_image, generation_info = self._backend.generate_image(
                    model_id=model_id,
                    prompt=enhanced_prompt,
                    original_prompts=valid_prompts,
                    combined_prompt=prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance_scale,
                    seed=actual_seed,
                    seed_original=seed_value,
                    seed_control=seed_control,
                    quantization=quantization,
                    t5_encoder=t5_encoder,
                    clip_encoder=clip_encoder,
                    loras=loras
                )
                
                # Validate generated image
                if generated_image is None:
                    raise ValueError("Backend returned None image")
                
                # Check for black frames
                import numpy as np
                img_array = np.array(generated_image)
                img_mean = np.mean(img_array)
                img_std = np.std(img_array)
                
                self.publish_update_to_parameter("status", f"🔍 Image stats: mean={img_mean:.2f}, std={img_std:.2f}")
                print(f"[FLUX DEBUG] Image stats: mean={img_mean:.2f}, std={img_std:.2f}")
                
                if img_mean < 10 and img_std < 5:
                    self.publish_update_to_parameter("status", f"⚠️ Generated image appears to be black/empty!")
                    print(f"[FLUX DEBUG] Black frame detected with {self._backend.get_name()} backend")
                
                # Save image using StaticFilesManager for proper UI rendering
                import io
                import hashlib
                import time
                from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
                
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                generated_image.save(img_buffer, format="PNG")
                image_bytes = img_buffer.getvalue()
                
                # Generate custom filename with workflow name and seed
                try:
                    # Try to get workflow name from Griptape context
                    workflow_name = "flux_generation"  # default fallback
                    
                    # Use seed and model for unique filename
                    model_short = model_id.split("/")[-1] if "/" in model_id else model_id
                    model_short = model_short.replace(".", "").replace("-", "_").lower()
                    
                    filename = f"{workflow_name}_{model_short}_seed_{actual_seed}.png"
                    
                    # Ensure filename is filesystem-safe
                    import re
                    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                    
                except Exception as e:
                    # Fallback to timestamp if custom naming fails
                    timestamp = int(time.time() * 1000)
                    filename = f"flux_mlx_seed_{actual_seed}_{timestamp}.png"
                
                print(f"[FLUX DEBUG] Custom filename: {filename}")
                
                # Save to managed static files and get URL for UI rendering
                try:
                    static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                        image_bytes, filename
                    )
                    self.publish_update_to_parameter("status", f"💾 Image saved: {filename}")
                    print(f"[FLUX DEBUG] Image saved: {static_url}")
                except Exception as save_error:
                    # Fallback: use temp file if StaticFilesManager fails
                    self.publish_update_to_parameter("status", f"⚠️ StaticFilesManager failed, using temp file: {str(save_error)}")
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        generated_image.save(tmp_file.name, format="PNG")
                        static_url = f"file://{tmp_file.name}"
                
                # Create ImageUrlArtifact
                try:
                    image_artifact = ImageUrlArtifact(value=static_url)
                    self.publish_update_to_parameter("status", f"✅ ImageUrlArtifact created successfully")
                except Exception as artifact_error:
                    # Try alternative constructor
                    image_artifact = ImageUrlArtifact(static_url)
                    self.publish_update_to_parameter("status", f"✅ ImageUrlArtifact created (fallback method)")
                
                # Add backend info to generation info
                generation_info["backend"] = self._backend.get_name()
                generation_info["enhanced_prompt"] = enhanced_prompt
                generation_info_str = json.dumps(generation_info, indent=2)
                
                # Set outputs
                self.parameter_output_values["image"] = image_artifact
                self.parameter_output_values["generation_info"] = generation_info_str
                
                # Set parameter outputs for workflow composition
                self.parameter_output_values["main_prompt"] = enhanced_prompt  # Output combined + enhanced prompt
                self.parameter_output_values["width"] = width
                self.parameter_output_values["height"] = height
                self.parameter_output_values["seed"] = actual_seed
                self.parameter_output_values["actual_seed"] = actual_seed
                
                model_display_name = self.flux_config.get_model_config(model_id).get('display_name', model_id)
                prompt_info = f"Prompts: {len(valid_prompts)}" if len(valid_prompts) > 1 else "Prompt: 1"
                final_status = f"✅ Generation complete!\nBackend: {self._backend.get_name()}\nModel: {model_display_name}\n{prompt_info}\nQuantization: {quantization}\nSeed: {actual_seed} ({seed_control})\nSize: {width}x{height}"
                self.publish_update_to_parameter("status", final_status)
                print(f"[FLUX DEBUG] ✅ Generation complete! Backend: {self._backend.get_name()}, {prompt_info}, Seed: {actual_seed} ({seed_control})")
                
                return final_status
                
            except Exception as e:
                error_msg = f"❌ Generation failed ({self._backend.get_name()}): {str(e)}"
                self.publish_update_to_parameter("status", error_msg)
                print(f"[FLUX DEBUG] {error_msg}")
                import traceback
                traceback.print_exc()
                # Set safe defaults
                self.parameter_output_values["image"] = None
                self.parameter_output_values["generation_info"] = "{}"
                self.parameter_output_values["main_prompt"] = ""
                self.parameter_output_values["width"] = 1024
                self.parameter_output_values["height"] = 1024
                self.parameter_output_values["seed"] = 12345
                self.parameter_output_values["actual_seed"] = 12345
                raise Exception(error_msg)

        # Return the generator for async processing
        yield generate_image 