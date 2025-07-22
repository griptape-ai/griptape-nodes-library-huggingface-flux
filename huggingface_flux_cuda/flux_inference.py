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
SERVICE = "Flux CUDA"

class FluxBackend(ABC):
    """Abstract base class for FLUX generation backends"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on current system"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get backend name for display"""
        pass
    
    @abstractmethod
    def generate_image(self, **kwargs) -> tuple[Any, dict]:
        """Generate image and return (pil_image, generation_info)"""
        pass
    
    @abstractmethod
    def validate_model_id(self, model_id: str) -> bool:
        """Check if model_id is supported by this backend"""
        pass



class DiffusersFluxBackend(FluxBackend):
    """Diffusers-based backend for CUDA/CPU"""
    
    def __init__(self):
        self._pipeline_cache: Dict[str, Any] = {}
    
    def is_available(self) -> bool:
        try:
            import torch
            from diffusers import FluxPipeline
            return True
        except ImportError:
            return False
    
    def get_name(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "Diffusers (CUDA)"
            else:
                return "Diffusers (CPU)"
        except ImportError:
            return "Diffusers"
    
    def validate_model_id(self, model_id: str) -> bool:
        return "FLUX.1-dev" in model_id or "FLUX.1-schnell" in model_id
    
    def _get_pipeline(self, model_id: str) -> Any:
        """Load diffusers pipeline with caching"""
        if model_id in self._pipeline_cache:
            return self._pipeline_cache[model_id]
        
        try:
            from diffusers import FluxPipeline
            import torch
        except ImportError:
            raise ImportError("Required packages not installed. Please install: pip install diffusers torch")
        
        # Load pipeline
        pipeline = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        
        # Move to appropriate device
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
        # Note: We avoid MPS for FLUX due to compatibility issues
        
        self._pipeline_cache[model_id] = pipeline
        return pipeline
    
    def generate_image(self, **kwargs) -> tuple[Any, dict]:
        import torch
        
        model_id = kwargs['model_id']
        prompt = kwargs['prompt']
        width = kwargs['width']
        height = kwargs['height']
        steps = kwargs['steps']
        guidance = kwargs['guidance']
        seed = kwargs['seed']
        max_sequence_length = kwargs.get('max_sequence_length', 512)
        
        pipeline = self._get_pipeline(model_id)
        
        # Set up generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": torch.Generator().manual_seed(seed)
        }
        
        # Add model-specific parameters
        if "FLUX.1-dev" in model_id:
            generation_kwargs["max_sequence_length"] = max_sequence_length
        
        # Generate image
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in cast")
            result = pipeline(**generation_kwargs)
            generated_image = result.images[0]
        
        generation_info = {
            "backend": "Diffusers",
            "model": model_id,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance,
            "seed": seed
        }
        
        if "FLUX.1-dev" in model_id:
            generation_info["max_sequence_length"] = max_sequence_length
        
        return generated_image, generation_info 

class FluxInference(ControlNode):
    """Unified FLUX inference node with automatic backend selection.
    
    Automatically chooses optimal backend:
    - MLX (Apple Silicon): Fast native generation using mflux
    - Diffusers (CUDA/CPU): Cross-platform fallback using diffusers
    """
    
    # Supported Flux models with their configurations
    FLUX_MODELS = {
        "black-forest-labs/FLUX.1-dev": {
            "display_name": "FLUX.1 Dev",
            "default_steps": 20,
            "max_steps": 50,
            "supports_guidance": True,
            "default_guidance": 7.5
        },
        "black-forest-labs/FLUX.1-schnell": {
            "display_name": "FLUX.1 Schnell", 
            "default_steps": 4,
            "max_steps": 8,
            "supports_guidance": False,
            "default_guidance": 1.0
        }
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Flux CUDA"
        
        # Auto-detect optimal backend
        self._backend = self._detect_optimal_backend()
        backend_name = self._backend.get_name()
        
        self.description = f"FLUX inference for CUDA/CPU systems using {backend_name}. Supports FLUX.1-dev and FLUX.1-schnell with automatic CUDA detection and CPU fallback."

        # Initialize available models by scanning cache
        available_models = self._scan_available_models()
        
        # Model Selection Group - Always visible
        with ParameterGroup(name=f"Model Selection ({backend_name})") as model_group:
            self.add_parameter(
                Parameter(
                    name="model",
                    tooltip="Flux model to use for generation. Models must be downloaded via HuggingFace nodes first.",
                    type="str",
                    default_value=available_models[0] if available_models else "",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=available_models)},
                    ui_options={"display_name": "Flux Model"}
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
            self.add_parameter(
                Parameter(
                    name="width",
                    tooltip="Width of generated image in pixels",
                    type="int",
                    default_value=1024,
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
                    default_value=1024,
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
                    default_value=7.5,
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

        # Model-Specific Groups for Diffusers backend
        self._create_model_specific_groups()
        
        # Set initial parameter group visibility based on default model
        if available_models:
            self._update_model_specific_visibility(available_models[0])

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

    def _detect_optimal_backend(self) -> FluxBackend:
        """Return Diffusers backend for CUDA/CPU systems"""
        diffusers_backend = DiffusersFluxBackend()
        if diffusers_backend.is_available():
            print(f"[FLUX INIT] Using Diffusers backend for CUDA/CPU")
            return diffusers_backend
        
        # Diffusers not available
        raise RuntimeError("Diffusers backend not available. This library requires 'diffusers' and 'torch' packages. For Apple Silicon, use the MLX library instead.")

    def _scan_available_models(self) -> list[str]:
        """Scan HuggingFace cache for available Flux models."""
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            return list(self.FLUX_MODELS.keys())  # Return all if HF not available
        
        available = []
        try:
            cache_info = scan_cache_dir()
            
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                if repo_id in self.FLUX_MODELS and self._backend.validate_model_id(repo_id):
                    # Verify the model appears complete (has some revisions)
                    if len(repo.revisions) > 0:
                        available.append(repo_id)
                        
        except Exception:
            # If scanning fails, return all supported models (user will get error if missing)
            available = [model_id for model_id in self.FLUX_MODELS.keys() if self._backend.validate_model_id(model_id)]
            
        # Always return at least one option, even if empty
        return available if available else [model_id for model_id in self.FLUX_MODELS.keys() if self._backend.validate_model_id(model_id)]

    def _create_model_specific_groups(self) -> None:
        """Create model-specific parameter groups for advanced Diffusers configuration"""
        
        # FLUX.1 Dev specific parameters
        with ParameterGroup(name="FLUX.1 Dev Settings") as dev_group:
            self.add_parameter(
                Parameter(
                    name="dev_num_inference_steps",
                    tooltip="Number of denoising steps for FLUX.1 Dev (more steps = higher quality, slower)",
                    type="int",
                    default_value=20,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Inference Steps"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="dev_max_sequence_length",
                    tooltip="Maximum sequence length for text encoder",
                    type="int",
                    default_value=512,
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=["256", "512", "1024"])},
                    ui_options={"display_name": "Max Sequence Length"}
                )
            )
        
        dev_group.ui_options = {"hide": True, "collapsed": True}
        self.add_node_element(dev_group)

        # FLUX.1 Schnell specific parameters  
        with ParameterGroup(name="FLUX.1 Schnell Settings") as schnell_group:
            self.add_parameter(
                Parameter(
                    name="schnell_num_inference_steps",
                    tooltip="Number of denoising steps for FLUX.1 Schnell (optimized for 1-4 steps)",
                    type="int", 
                    default_value=4,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Inference Steps"}
                )
            )
        
        schnell_group.ui_options = {"hide": True, "collapsed": True}
        self.add_node_element(schnell_group)

    def after_value_set(self, parameter: Parameter, value: Any, modified_parameters_set: set[str] | None = None) -> None:
        """Handle parameter changes, especially model selection."""
        if parameter.name == "model":
            self._update_model_specific_visibility(value)
            self._update_guidance_scale_for_model(value)

    def _update_model_specific_visibility(self, selected_model: str) -> None:
        """Show/hide model-specific parameter groups based on selection."""
        # Hide all model-specific groups first by hiding individual parameters
        self.hide_parameter_by_name("dev_num_inference_steps")
        self.hide_parameter_by_name("dev_max_sequence_length")
        self.hide_parameter_by_name("schnell_num_inference_steps")
        
        # Show relevant parameters based on model
        if "FLUX.1-dev" in selected_model:
            self.show_parameter_by_name("dev_num_inference_steps")
            self.show_parameter_by_name("dev_max_sequence_length")
        elif "FLUX.1-schnell" in selected_model:
            self.show_parameter_by_name("schnell_num_inference_steps")

    def _update_guidance_scale_for_model(self, selected_model: str) -> None:
        """Update guidance scale based on model capabilities."""
        model_config = self.FLUX_MODELS.get(selected_model, {})
        
        if not model_config.get("supports_guidance", True):
            # For models that don't support guidance (like Schnell), set to 1.0
            self.parameter_output_values["guidance_scale"] = 1.0

    def process(self) -> AsyncResult:
        """Generate image using optimal backend."""
        
        def generate_image() -> str:
            try:
                # Get parameters
                model_id = self.get_parameter_value("model")
                prompt = self.get_parameter_value("prompt")
                width = int(self.get_parameter_value("width"))
                height = int(self.get_parameter_value("height"))
                guidance_scale = float(self.get_parameter_value("guidance_scale"))
                seed = int(self.get_parameter_value("seed"))
                
                # Validate inputs
                if not model_id:
                    raise ValueError("No Flux model selected. Please select a model.")
                if not prompt or not prompt.strip():
                    raise ValueError("Prompt is required for image generation.")
                
                # Validate model compatibility with backend
                if not self._backend.validate_model_id(model_id):
                    raise ValueError(f"Model {model_id} not supported by {self._backend.get_name()} backend")
                
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
                
                # Get model-specific parameters from Diffusers backend
                if "FLUX.1-dev" in model_id:
                    num_steps = int(self.get_parameter_value("dev_num_inference_steps"))
                    max_sequence_length = int(self.get_parameter_value("dev_max_sequence_length"))
                    # Ensure minimum values
                    num_steps = max(20, num_steps)
                    max_sequence_length = max(512, max_sequence_length)
                elif "FLUX.1-schnell" in model_id:
                    num_steps = int(self.get_parameter_value("schnell_num_inference_steps"))
                    max_sequence_length = 512
                    # Ensure valid range for schnell
                    num_steps = max(1, min(4, num_steps))
                    guidance_scale = 1.0  # Force correct guidance for schnell
                else:
                    num_steps = 20
                    max_sequence_length = 512
                
                self.publish_update_to_parameter("status", f"🚀 Generating with {self._backend.get_name()} backend...")
                self.publish_update_to_parameter("status", f"📝 Prompt: '{enhanced_prompt[:50]}{'...' if len(enhanced_prompt) > 50 else ''}'")
                self.publish_update_to_parameter("status", f"⚙️ Settings: {width}x{height}, {num_steps} steps, guidance={guidance_scale}")
                print(f"[FLUX DEBUG] Backend: {self._backend.get_name()}")
                print(f"[FLUX DEBUG] Model: {model_id}")
                print(f"[FLUX DEBUG] Enhanced prompt: '{enhanced_prompt}'")
                
                # Generate using backend
                generated_image, generation_info = self._backend.generate_image(
                    model_id=model_id,
                    prompt=enhanced_prompt,
                    width=width,
                    height=height,
                    steps=num_steps,
                    guidance=guidance_scale,
                    seed=seed,
                    max_sequence_length=max_sequence_length
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
                filename = f"flux_{self._backend.get_name().lower().replace(' ', '_')}_{timestamp}_{content_hash}.png"
                
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
                
                final_status = f"✅ Generation complete!\nBackend: {self._backend.get_name()}\nModel: {self.FLUX_MODELS.get(model_id, {}).get('display_name', model_id)}\nSeed: {seed}\nSize: {width}x{height}"
                self.publish_update_to_parameter("status", final_status)
                print(f"[FLUX DEBUG] ✅ Generation complete! Backend: {self._backend.get_name()}, Seed: {seed}")
                
                return final_status
                
            except Exception as e:
                error_msg = f"❌ Generation failed ({self._backend.get_name()}): {str(e)}"
                self.publish_update_to_parameter("status", error_msg)
                print(f"[FLUX DEBUG] {error_msg}")
                # Set safe defaults
                self.parameter_output_values["image"] = None
                self.parameter_output_values["generation_info"] = "{}"
                raise Exception(error_msg)

        # Return the generator for async processing
        yield generate_image 