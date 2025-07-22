import json
import os
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Protocol
from abc import ABC, abstractmethod
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.traits.options import Options

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
        """Scan HuggingFace cache for available T5 encoders"""
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            # If HF not available, return defaults
            return [self.config["global_defaults"]["default_t5_encoder"]]
        
        available_t5 = []
        
        try:
            cache_info = scan_cache_dir()
            
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                
                # Check if this looks like a T5 model
                if self._is_t5_encoder(repo_id):
                    available_t5.append(repo_id)
                    print(f"[FLUX T5] Found T5 encoder: {repo_id}")
                    
        except Exception as e:
            print(f"[FLUX T5] Error scanning cache: {e}")
            
        # Always include default as fallback
        default_t5 = self.config["global_defaults"]["default_t5_encoder"]
        if default_t5 not in available_t5:
            available_t5.insert(0, default_t5)
            
        return available_t5 if available_t5 else [default_t5]
    
    def get_recommended_t5_encoder(self, model_id: str) -> str:
        """Get recommended T5 encoder for a specific model"""
        model_config = self.get_model_config(model_id)
        return model_config.get("recommended_t5_encoder", 
                              self.config["global_defaults"]["default_t5_encoder"])
    
    def _is_t5_encoder(self, repo_id: str) -> bool:
        """Check if repo_id appears to be a T5 encoder model"""
        repo_lower = repo_id.lower()
        
        # T5 patterns
        t5_patterns = [
            "t5" in repo_lower and ("google" in repo_lower or "huggingface" in repo_lower),
            "t5-" in repo_lower,
            "flan-t5" in repo_lower,
            repo_lower.startswith("google/t5"),
            repo_lower.startswith("google/flan-t5")
        ]
        
        return any(t5_patterns)

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
            from mflux.generate import Flux1, Config
            import mlx.core as mx
            
        except ImportError as e:
            raise ImportError("mflux not installed. Run: pip install mflux")
        
        model_id = kwargs['model_id']
        prompt = kwargs['prompt']
        width = kwargs['width']
        height = kwargs['height']
        steps = kwargs['steps']
        guidance = kwargs['guidance']
        seed = kwargs['seed']
        quantization = kwargs.get('quantization', 4)
        t5_encoder = kwargs.get('t5_encoder', self.config.config["global_defaults"]["default_t5_encoder"])
        
        # Get model config (with global defaults if needed)
        model_config = self.config.get_model_config(model_id)
        mflux_model = model_config['mflux_name']
        
        # Debug logging for encoder selection  
        print(f"[FLUX T5] Selected T5 encoder: {t5_encoder}")
        print(f"[FLUX T5] NOTE: mflux currently does not support custom encoders")
        print(f"[FLUX T5] The model will use default encoders regardless of selection")
        print(f"[FLUX T5] This parameter is prepared for future mflux encoder support")
        
        # Create Flux1 model with quantization
        quantize_param = None if quantization == "none" else int(quantization.replace("-bit", ""))
        flux_model = Flux1.from_name(mflux_model, quantize=quantize_param)
        
        # Create generation config
        config = Config(
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance=guidance if model_config['supports_guidance'] else 1.0
        )
        
        # Generate image
        result = flux_model.generate_image(
            seed=seed,
            prompt=prompt,
            config=config
        )
        
        # Extract PIL image from result
        image = result.image
        
        generation_info = {
            "backend": "MLX",
            "model": model_id,
            "mflux_model": mflux_model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "quantization": quantization,
            "t5_encoder": t5_encoder
        }
        
        return image, generation_info

class FluxInference(ControlNode):
    """FLUX inference node optimized for Apple Silicon using MLX"""
    
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
                    name="prompt",
                    tooltip="Text description of the image to generate",
                    type="str",
                    input_types=["str"],
                    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                    ui_options={
                        "multiline": True,
                        "placeholder_text": "Describe the image you want to generate...",
                        "display_name": "Prompt"
                    }
                )
            )
        
        model_group.ui_options = {"collapsed": False}
        self.add_node_element(model_group)

        # Shared Generation Settings - Always visible  
        with ParameterGroup(name="Generation Settings") as gen_group:
            global_defaults = self.flux_config.config["global_defaults"]
            
            self.add_parameter(
                Parameter(
                    name="width",
                    tooltip="Width of generated image in pixels",
                    type="int",
                    default_value=global_defaults["default_width"],
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=["512", "768", "1024", "1152", "1280"])},
                    ui_options={"display_name": "Width"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="height", 
                    tooltip="Height of generated image in pixels",
                    type="int",
                    default_value=global_defaults["default_height"],
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=["512", "768", "1024", "1152", "1280"])},
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
            
            self.add_parameter(
                Parameter(
                    name="seed",
                    tooltip="Random seed for reproducible generation (-1 for random)",
                    type="int", 
                    default_value=-1,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Seed"}
                )
            )
        
        gen_group.ui_options = {"collapsed": False}
        self.add_node_element(gen_group)

        # Advanced Configuration Group - Collapsed by default
        with ParameterGroup(name="Advanced Configuration") as advanced_group:
            # Advanced Configuration Toggle
            self.add_parameter(
                Parameter(
                    name="advanced_config",
                    tooltip="Show advanced encoder configuration options",
                    type="bool",
                    default_value=False,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Show Advanced Options"}
                )
            )
            
            # Get available T5 encoders
            available_t5_encoders = self.flux_config.get_available_t5_encoders()
            default_t5 = self.flux_config.get_recommended_t5_encoder(available_models[0] if available_models else "")
            
            self.add_parameter(
                Parameter(
                    name="t5_encoder",
                    tooltip="⚠️ Future Feature: T5 text encoder selection. Currently mflux uses default encoders only. This parameter is prepared for when mflux adds custom encoder support.",
                    type="str",
                    default_value=default_t5,
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=available_t5_encoders)},
                    ui_options={"display_name": "T5 Text Encoder (Future)", "hide": True}  # Hidden by default
                )
            )

        advanced_group.ui_options = {"collapsed": True}
        self.add_node_element(advanced_group)

        # Set initial quantization parameter options based on default model
        if available_models:
            self._update_quantization_options(available_models[0])

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

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Callback triggered when parameter values change"""        
        if parameter.name == "model":
            # Update quantization parameter options when model changes
            self._update_quantization_options(value)
            # Update T5 encoder recommendation when model changes (only if advanced config is enabled)
            if self.get_parameter_value("advanced_config"):
                self._update_t5_encoder_recommendation(value)
        elif parameter.name == "advanced_config":
            # Show/hide T5 encoder parameter
            self._toggle_t5_visibility(value)
    
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
    
    def _toggle_t5_visibility(self, show_t5: bool) -> None:
        """Show or hide the T5 encoder parameter"""
        t5_param = self.get_parameter_by_name("t5_encoder")
        if t5_param:
            ui_options = t5_param.ui_options.copy()
            ui_options["hide"] = not show_t5
            t5_param.ui_options = ui_options
            print(f"[FLUX] T5 encoder parameter {'shown' if show_t5 else 'hidden'}")
    


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

    def process(self) -> AsyncResult:
        """Generate image using MLX backend."""
        
        def generate_image() -> str:
            try:
                # Get parameters
                model_id = self.get_parameter_value("model")
                prompt = self.get_parameter_value("prompt")
                width = int(self.get_parameter_value("width"))
                height = int(self.get_parameter_value("height"))
                guidance_scale = float(self.get_parameter_value("guidance_scale"))
                seed = int(self.get_parameter_value("seed"))
                quantization = self.get_parameter_value("quantization")
                t5_encoder = self.get_parameter_value("t5_encoder")
                
                # Validate inputs
                if not model_id:
                    raise ValueError("No Flux model selected. Please select a model.")
                if not prompt or not prompt.strip():
                    raise ValueError("Prompt is required for image generation.")
                
                # Validate model compatibility with backend
                if not self._backend.validate_model_id(model_id):
                    raise ValueError(f"Model {model_id} not supported by {self._backend.get_name()} backend")
                
                # Validate quantization settings for pre-quantized models
                is_pre_quantized, quant_type, warning = self.flux_config.is_model_pre_quantized(model_id)
                if is_pre_quantized and quantization != "none":
                    self.publish_update_to_parameter("status", f"⚠️ Warning: Model is pre-quantized ({quant_type}), forcing quantization to 'none'")
                    print(f"[FLUX DEBUG] Pre-quantized model detected, overriding quantization: {quantization} → none")
                    quantization = "none"
                
                # Handle random seed
                if seed == -1:
                    import random
                    seed = random.randint(0, 2**32 - 1)
                
                # Enhance simple prompts for better generation
                enhanced_prompt = prompt.strip()
                if len(enhanced_prompt.split()) < 3:
                    if enhanced_prompt.lower() == "capybara":
                        enhanced_prompt = "a cute capybara sitting on grass, detailed, high quality, photorealistic"
                        enhance_msg = f"📝 Enhanced simple prompt: '{prompt.strip()}' → '{enhanced_prompt}'"
                        self.publish_update_to_parameter("status", enhance_msg)
                        print(f"[FLUX DEBUG] {enhance_msg}")
                
                # Get model-specific settings from config (with global defaults fallback)
                model_config = self.flux_config.get_model_config(model_id)
                num_steps = model_config['default_steps']
                
                # Override guidance for schnell model
                if not model_config['supports_guidance']:
                    guidance_scale = 1.0
                
                self.publish_update_to_parameter("status", f"🚀 Generating with {self._backend.get_name()} backend...")
                self.publish_update_to_parameter("status", f"📝 Prompt: '{enhanced_prompt[:50]}{'...' if len(enhanced_prompt) > 50 else ''}'")
                self.publish_update_to_parameter("status", f"⚙️ Settings: {width}x{height}, {num_steps} steps, guidance={guidance_scale}, quantization={quantization}")
                print(f"[FLUX DEBUG] Backend: {self._backend.get_name()}")
                print(f"[FLUX DEBUG] Model: {model_id}")
                print(f"[FLUX DEBUG] Enhanced prompt: '{enhanced_prompt}'")
                print(f"[FLUX DEBUG] Quantization: {quantization}")
                print(f"[FLUX DEBUG] T5 Encoder: {t5_encoder}")
                print(f"[FLUX DEBUG] Model config: {model_config}")
                
                # Generate using backend
                generated_image, generation_info = self._backend.generate_image(
                    model_id=model_id,
                    prompt=enhanced_prompt,
                    width=width,
                    height=height,
                    steps=num_steps,
                    guidance=guidance_scale,
                    seed=seed,
                    quantization=quantization,
                    t5_encoder=t5_encoder
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
                
                # Generate unique filename with timestamp and hash
                timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
                content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                filename = f"flux_mlx_{timestamp}_{content_hash}.png"
                
                # Save to managed static files and get URL for UI rendering
                try:
                    static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                        image_bytes, filename
                    )
                    self.publish_update_to_parameter("status", f"✅ Image saved: {filename}")
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
                
                model_display_name = self.flux_config.get_model_config(model_id).get('display_name', model_id)
                final_status = f"✅ Generation complete!\nBackend: {self._backend.get_name()}\nModel: {model_display_name}\nQuantization: {quantization}\nSeed: {seed}\nSize: {width}x{height}"
                self.publish_update_to_parameter("status", final_status)
                print(f"[FLUX DEBUG] ✅ Generation complete! Backend: {self._backend.get_name()}, Seed: {seed}")
                
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
                raise Exception(error_msg)

        # Return the generator for async processing
        yield generate_image 