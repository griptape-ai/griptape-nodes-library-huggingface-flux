import json
import os
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Protocol
import subprocess, sys

from abc import ABC, abstractmethod
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup, ParameterList
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.traits.options import Options
from .flux_config import FluxConfig
from .flux_cuda_helpers import PromptBuilder, ImageInputManager, create_error_image, SeedManager

# Import shared backend - handle both package and direct loading contexts
def get_shared_backend():
    """Get the shared backend from the advanced library"""
    try:
        # Try relative import first (when loaded as package)
        print("[FLUX DEBUG] Trying relative import...")
        from .. import get_shared_backend as _get_shared_backend
        result = _get_shared_backend()
        print("[FLUX DEBUG] ✅ Relative import successful")
        return result
    except ImportError as e1:
        print(f"[FLUX DEBUG] Relative import failed: {e1}")
        try:
            # Method 1: Direct import from advanced library
            print("[FLUX DEBUG] Trying direct advanced library import...")
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            print(f"[FLUX DEBUG] Parent dir: {parent_dir}")
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from library_loader import get_shared_backend as _get_shared_backend
            result = _get_shared_backend()
            print("[FLUX DEBUG] ✅ Direct advanced library import successful")
            return result
        except ImportError as e2:
            print(f"[FLUX DEBUG] Direct advanced library import failed: {e2}")
            try:
                # Method 2: Try absolute import 
                print("[FLUX DEBUG] Trying absolute import...")
                from huggingface_cuda import get_shared_backend as _get_shared_backend
                result = _get_shared_backend()
                print("[FLUX DEBUG] ✅ Absolute import successful")
                return result
            except ImportError as e3:
                print(f"[FLUX DEBUG] Absolute import failed: {e3}")
                try:
                    # Method 3: Direct module loading
                    print("[FLUX DEBUG] Trying direct module loading...")
                    import importlib.util
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    advanced_path = os.path.join(parent_dir, "library_loader.py")
                    print(f"[FLUX DEBUG] Library loader path: {advanced_path}")
                    print(f"[FLUX DEBUG] Path exists: {os.path.exists(advanced_path)}")
                    spec = importlib.util.spec_from_file_location("library_loader", advanced_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    result = module.get_shared_backend()
                    print("[FLUX DEBUG] ✅ Direct module loading successful")
                    return result
                except Exception as e4:
                    # Final fallback
                    print(f"[FLUX DEBUG] ❌ All shared backend access methods failed: {e4}")
                    return {"available": False, "error": f"Shared backend not accessible: {e4}"}

# -----------------------------------------------------------------------------
# bitsandbytes compatibility helper
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Service configuration for environment variables access
SERVICE = "HuggingFace CUDA"
API_KEY_ENV_VAR = "HUGGINGFACE_HUB_ACCESS_TOKEN"

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
    def generate_image(self, **kwargs):
        """Generate image and return (pil_image, generation_info) - can be generator"""
        pass
    
    @abstractmethod
    def validate_model_id(self, model_id: str) -> bool:
        """Check if model_id is supported by this backend"""
        pass



class DeferredFluxBackend(FluxBackend):
    """Deferred backend that attempts initialization when first used"""
    
    def __init__(self):
        self._actual_backend = None
        self._initialization_attempted = False
        self._initialization_error = None
    
    def _ensure_backend(self):
        """Attempt to initialize the actual backend on first use"""
        if self._initialization_attempted:
            if self._actual_backend is None:
                raise RuntimeError(f"Backend initialization failed: {self._initialization_error}")
            return self._actual_backend

        self._initialization_attempted = True
        try:
            print("[FLUX DEBUG] Attempting deferred backend initialization...")
            from .flux_cuda_backend import get_cuda_backend
            backend = get_cuda_backend()
            if backend.is_available():
                self._actual_backend = backend
                print("[FLUX DEBUG] ✅ Deferred backend initialization successful")
                return backend
            else:
                raise RuntimeError("Backend not available after initialization attempt")
        except Exception as e:
            self._initialization_error = str(e)
            error_msg = f"Failed to initialize Diffusers backend: {e}. This library requires 'torch' and 'diffusers' packages. Dependencies should be automatically installed by Griptape."
            raise RuntimeError(error_msg)
    
    def is_available(self) -> bool:
        """Always return True during node creation - actual check happens during use"""
        return True
    
    def get_name(self) -> str:
        """Return name without requiring initialization"""
        return "Diffusers (Deferred Loading)"
    
    def validate_model_id(self, model_id: str) -> bool:
        """Basic validation without requiring backend"""
        return "FLUX.1-dev" in model_id or "FLUX.1-schnell" in model_id
    
    def generate_image(self, **kwargs):
        """Delegate to actual backend, initializing if needed"""
        print(f"[FLUX DEBUG] ===== DeferredFluxBackend.generate_image CALLED =====")
        print(f"[FLUX DEBUG] Quantization from kwargs: {kwargs.get('quantization', 'NOT_FOUND')}")
        
        backend = self._ensure_backend()
        print(f"[FLUX DEBUG] Actual backend after ensure: {type(backend)} - {backend.__class__.__name__}")
        
        # Forward to the actual backend's generate_image method directly
        print(f"[FLUX DEBUG] Calling actual backend generate_image...")
        backend_result = backend.generate_image(**kwargs)
        
        print(f"[FLUX DEBUG] Backend result type: {type(backend_result)}")
        print(f"[FLUX DEBUG] Returning backend result directly")
        return backend_result

class FluxInference(ControlNode):
    """Unified FLUX inference node with automatic backend selection.

    Dynamically adds a `control_images` input when the selected model supports
    image conditioning (e.g., Kontext / Fill variants). This keeps the UI clean
    for text-only models.
    """

    _image_parameter_template = ParameterList(
        name="control_images",
        input_types=["ImageUrlArtifact", "list[ImageUrlArtifact]"],
        default_value=[],
        tooltip="Optional control or conditioning images (only for models that support it)",
        allowed_modes={ParameterMode.INPUT},
        ui_options={"display_name": "Control Images"}
    )
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
        import time
        init_start = time.time()
        print(f"[FLUX INIT] Starting FluxInference initialization...")
        
        super().__init__(**kwargs)

        # Load config file (image-input capabilities)
        self._flux_cfg = FluxConfig(Path(__file__).with_name("flux_config.json"))
        # Image input manager
        self._img_mgr = ImageInputManager(self, self._flux_cfg)
        self.category = "Flux CUDA"
        
        # Use pre-loaded shared backend (fast!)
        backend_start = time.time()
        print(f"[FLUX INIT] Using shared CUDA backend...")
        from .flux_cuda_backend import get_cuda_backend
        self._backend = get_cuda_backend()
        backend_name = self._backend.get_name()
        backend_time = time.time() - backend_start
        print(f"[FLUX INIT] Backend ready in {backend_time:.3f}s -> {backend_name}")
        
        # Dynamic description based on backend type
        if "Deferred" in backend_name:
            self.description = f"FLUX inference for CUDA/CPU systems. Dependencies will be loaded automatically when needed. Supports FLUX.1-dev and FLUX.1-schnell with automatic CUDA detection and CPU fallback."
        else:
            self.description = f"FLUX inference for CUDA/CPU systems using {backend_name}. Supports FLUX.1-dev and FLUX.1-schnell with automatic CUDA detection and CPU fallback."

        # Initialize available models by scanning cache
        scan_start = time.time()
        print(f"[FLUX INIT] Scanning available models...")
        available_models = self._scan_available_models()
        scan_time = time.time() - scan_start
        print(f"[FLUX INIT] Model scanning completed in {scan_time:.2f}s -> Found {len(available_models)} models: {available_models}")
        
        # Model Selection Group - Always visible
        with ParameterGroup(name=f"Model Selection ({backend_name})") as model_group:
            
            
            # Main prompt (single string)
            self.add_parameter(
                Parameter(
                    name="main_prompt",
                    tooltip="Primary text description for image generation",
                    type="str",
                    input_types=["str"],
                    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                    ui_options={
                        "display_name": "Main Prompt",
                        "multiline": True,
                        "placeholder_text": "Describe the image you want to generate..."
                    }
                )
            )
            # Additional prompts (list input)
            self.add_parameter(
                ParameterList(
                    name="additional_prompts",
                    input_types=["str", "list[str]"],
                    default_value=[],
                    tooltip="Optional additional prompts to merge with the main prompt.",
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": "Additional Prompts"}
                )
            )
            
            self.add_parameter(
                Parameter(
                    name="system_constraints",
                    tooltip="Optional system constraints from GPU Configuration node. Supports: 'gpu_memory_gb' (general limit), '4-bit_memory_gb'/'8-bit_memory_gb'/'none_memory_gb' (quantization-specific limits), 'allow_low_memory' (bypass safety checks), 'gpu_device' (device selection).",
                    type="dict",
                    input_types=["dict"],
                    default_value={},
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": "GPU Configuration",  "hide_property": True}
                )
            )
            self.add_parameter(
                Parameter(
                    name="flux_config",
                    tooltip="Flux configuration dict from selector",
                    type="dict",
                    input_types=["dict"],
                    default_value={},
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": "Flux Model", "hide_property": True}
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
                    traits={Options(choices=[512, 768, 1024, 1152, 1280])},
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
                    traits={Options(choices=[512, 768, 1024, 1152, 1280])},
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
                    name="steps",
                    tooltip="Number of inference steps. More steps = higher quality but slower generation. FLUX.1-dev: 15-50 steps, FLUX.1-schnell: 1-8 steps recommended.",
                    type="int",
                    default_value=20,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Inference Steps"}
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
            

            # Scheduler / sampler parameters
            self.add_parameter(
                Parameter(
                    name="scheduler",
                    tooltip="Sampling scheduler",
                    type="str",
                    default_value="DPM++ 2M Karras",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=[
                        "Euler A",
                        "Euler",
                        "DDIM",
                        "DPM++ 2M Karras",
                        "FlowMatchEuler"
                    ])},
                    ui_options={"display_name": "Scheduler"}
                )
            )
            self.add_parameter(
                Parameter(
                    name="cfg_rescale",
                    tooltip="CFG rescale (0 disables)",
                    type="float",
                    default_value=0.0,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "CFG Rescale"}
                )
            )
            self.add_parameter(
                Parameter(
                    name="denoise_eta",
                    tooltip="DDIM / noise eta",
                    type="float",
                    default_value=0.0,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Denoise Eta"}
                )
            )
        
        gen_group.ui_options = {"collapsed": False}
        self.add_node_element(gen_group)

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
        
        # Log total initialization time
        total_init_time = time.time() - init_start
        print(f"[FLUX INIT] ✅ FluxInference initialization completed in {total_init_time:.2f}s")

    def _scan_available_models(self) -> list[str]:
        """Dynamically scan HuggingFace cache for FLUX models by analyzing model structure."""
        try:
            from huggingface_hub import scan_cache_dir
            import json
            from pathlib import Path
        except ImportError:
            # If HF not available, return config models as fallback
            available = list(self.FLUX_MODELS.keys())
            print(f"[FLUX SCAN] HuggingFace hub not available, using defaults: {available}")
            return available
        
        available = []
        try:
            cache_info = scan_cache_dir()
            
            # Handle different HF hub API versions
            cache_dir = getattr(cache_info, 'cache_dir', getattr(cache_info, 'cache_path', 'unknown'))
            print(f"[FLUX SCAN] Scanning cache directory: {cache_dir}")
            
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                
                # First check if it has FLUX in the name at all
                if not self._is_flux_model(repo_id):
                    continue
                
                # Now analyze the actual model structure
                if len(repo.revisions) > 0:
                    # Get the latest revision's snapshot path
                    latest_revision = next(iter(repo.revisions))
                    snapshot_path = Path(latest_revision.snapshot_path)
                    
                    if self._is_base_flux_model(snapshot_path, repo_id):
                        # Critical: Check if model is actually loadable
                        if self._is_model_loadable(repo_id):
                            available.append(repo_id)
                            print(f"[FLUX SCAN] ✅ Found complete FLUX model: {repo_id}")
                        else:
                            print(f"[FLUX SCAN] ❌ Skipping incomplete FLUX model: {repo_id}")
                    else:
                        print(f"[FLUX SCAN] Skipping specialized FLUX model: {repo_id}")
                        
        except Exception as e:
            print(f"[FLUX SCAN] Error scanning cache: {e}")
            # If scanning fails, return config models as fallback
            available = list(self.FLUX_MODELS.keys())
            
        # Always return at least one option (fallback to config models)
        if not available:
            available = list(self.FLUX_MODELS.keys())
            print("[FLUX SCAN] No models found in cache, using config defaults")
        
        print(f"[FLUX SCAN] Available models: {available}")
        return available

    def _is_flux_model(self, model_id: str) -> bool:
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
        ]
        
        return any(flux_patterns)

    def _is_base_flux_model(self, snapshot_path, repo_id: str) -> bool:
        """Analyze model structure to determine if it's a base FLUX text-to-image model."""
        try:
            from pathlib import Path
            import json
            
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

    def _is_encoder_only_repository(self, snapshot_path, repo_id: str) -> bool:
        """Check if this repository only contains text encoders (T5/CLIP)"""
        try:
            from pathlib import Path
            
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

    def _is_model_loadable(self, model_id: str) -> bool:
        """Test if a model can actually be loaded without errors."""
        try:
            # Get shared backend modules
            from huggingface_hub import snapshot_download
            import os
            
            # First check: Try to get model info without downloading
            try:
                # Use offline mode to check if model is already fully cached
                snapshot_path = snapshot_download(
                    repo_id=model_id,
                    local_files_only=True,  # Only use local cache
                    allow_patterns=["config.json", "model_index.json", "*.safetensors"]
                )
                
                # Check if essential files exist
                from pathlib import Path
                snapshot_path = Path(snapshot_path)
                
                # For FLUX models, we need either model_index.json (pipeline) or config.json (transformer)
                has_model_index = (snapshot_path / "model_index.json").exists()
                has_config = (snapshot_path / "config.json").exists()
                
                if not (has_model_index or has_config):
                    print(f"[FLUX SCAN] {model_id}: Missing essential config files")
                    return False
                
                # Check for model weight files
                safetensors_files = list(snapshot_path.rglob("*.safetensors"))
                if not safetensors_files:
                    print(f"[FLUX SCAN] {model_id}: No model weight files found")
                    return False
                
                # Quick validation: try to load config without actually loading weights
                if has_model_index:
                    import json
                    with open(snapshot_path / "model_index.json") as f:
                        model_index = json.load(f)
                    if not model_index.get("_class_name"):
                        print(f"[FLUX SCAN] {model_id}: Invalid model_index.json")
                        return False
                
                print(f"[FLUX SCAN] {model_id}: Validation passed - model is loadable")
                return True
                
            except Exception as e:
                print(f"[FLUX SCAN] {model_id}: Not fully cached or corrupted - {e}")
                return False
                
        except Exception as e:
            print(f"[FLUX SCAN] {model_id}: Error during loadability check - {e}")
            return False

    def after_value_set(self, parameter: Parameter, value: Any, modified_parameters_set: set[str] | None = None) -> None:
        # Inject / remove control_images parameter when model selection changes
        if parameter.name == "flux_config":
            model_id = value.get("model_id") if isinstance(value, dict) else ""
            self._img_mgr.sync(model_id)

        """Handle parameter changes, especially model selection."""
        if parameter.name == "model":  # retained for legacy flows
            self._update_guidance_scale_for_model(value)

    def _update_guidance_scale_for_model(self, selected_model: str) -> None:
        """Update guidance scale based on model capabilities."""
        model_config = self.FLUX_MODELS.get(selected_model, {})

    def process(self) -> AsyncResult[None]:
        """Generate image using optimal backend."""
        
        # Use the working yield pattern - yield function to call asynchronously
        yield lambda: self._process()
    
    def _process(self):
        # ---------- collect UI parameters ----------
        flux_cfg          = self.get_parameter_value("flux_config") or {}
        model_id          = flux_cfg.get("model_id", "")
        main_prompt       = self.get_parameter_value("main_prompt")
        extra_prompts     = self.get_parameter_list_value("additional_prompts")

        # merge prompts via helper
        prompt, valid_prompts = PromptBuilder.combine(main_prompt, extra_prompts)

        # optional image inputs
        control_images    = self._img_mgr.get_images(model_id)

        width             = int(self.get_parameter_value("width"))
        height            = int(self.get_parameter_value("height"))
        guidance_scale    = float(self.get_parameter_value("guidance_scale"))
        seed              = int(self.get_parameter_value("seed"))
        scheduler_name    = self.get_parameter_value("scheduler")
        cfg_rescale       = float(self.get_parameter_value("cfg_rescale"))
        denoise_eta       = float(self.get_parameter_value("denoise_eta"))
        quantization      = flux_cfg.get("quantization", "none")
        system_constraints = self.get_parameter_value("system_constraints") or {}

        # ---------- basic validation ----------
        if not model_id:
            raise ValueError("No Flux model selected.")
        if not prompt.strip():
            raise ValueError("Prompt is required for image generation.")
        if not self._backend.validate_model_id(model_id):
            raise ValueError(f"{model_id} not supported by {self._backend.get_name()}")

        # ---------- seed handling ----------
        if seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)

        # ---------- backend call ----------
        backend_result = self._backend.generate_image(
            model_id=model_id,
            prompt=prompt,
            width=width,
            height=height,
            steps=int(self.get_parameter_value("steps")),
            guidance=guidance_scale,
            seed=seed,
            quantization=quantization,
            scheduler=scheduler_name,
            cfg_rescale=cfg_rescale,
            denoise_eta=denoise_eta,
            system_constraints=system_constraints,
            **({"control_images": control_images} if control_images else {})
        )

        steps=int(self.get_parameter_value("steps"))
        num_steps=steps          # keep the old variable name used later
        
        # Process backend result
        if isinstance(backend_result, tuple) and len(backend_result) == 2:
            generated_image, generation_info = backend_result
            print(f"[FLUX DEBUG] Got tuple result: image={type(generated_image)}, info={type(generation_info)}")
        else:
            print(f"[FLUX DEBUG] Backend result format: {type(backend_result)}")
            # Handle FluxPipelineOutput from diffusers
            if hasattr(backend_result, 'images') and backend_result.images:
                generated_image = backend_result.images[0]  # First image
                generation_info = {
                    "backend": self._backend.get_name(),
                    "model": model_id,
                    "actual_seed": seed,
                    "steps": num_steps,
                    "guidance": guidance_scale
                }
                print(f"[FLUX DEBUG] Extracted from FluxPipelineOutput: image={type(generated_image)}")
            else:
                print(f"[FLUX DEBUG] Unexpected backend result format: {type(backend_result)}")
                raise RuntimeError(f"Backend returned unexpected format: {type(backend_result)}")
        
        print(f"[FLUX DEBUG] ===== PROCESSING GENERATION RESULT =====")
        print(f"[FLUX DEBUG] Generated image type: {type(generated_image)}")
        print(f"[FLUX DEBUG] Generation info type: {type(generation_info)}")
        print(f"[FLUX DEBUG] Generation info keys: {generation_info.keys() if isinstance(generation_info, dict) else 'NOT_DICT'}")
        
        # Ensure we have a PIL Image
        if not hasattr(generated_image, 'save'):
            print(f"[FLUX DEBUG] Converting result to PIL Image...")
            try:
                from PIL import Image
                if hasattr(generated_image, 'numpy'):  # torch tensor
                    import numpy as np
                    img_array = generated_image.cpu().numpy()
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype('uint8')
                    generated_image = Image.fromarray(img_array)
                else:
                    raise RuntimeError(f"Cannot convert {type(generated_image)} to PIL Image")
            except Exception as conv_error:
                print(f"[FLUX DEBUG] Conversion failed: {conv_error}")
                raise RuntimeError(f"Failed to convert generated image to PIL format: {conv_error}")
        
        print(f"[FLUX DEBUG] Final image type: {type(generated_image)}")
        print(f"[FLUX DEBUG] Final image size: {generated_image.size if hasattr(generated_image, 'size') else 'NO_SIZE'}")
        
        # Save the image and create URL
        import io
        import hashlib
        import time
        
        # Convert image to bytes
        image_bytes = io.BytesIO()
        generated_image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()
        
        # Generate unique filename
        content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        timestamp = int(time.time() * 1000)
        filename = f"flux_generated_{timestamp}_{content_hash}.png"
        
        # Save using GriptapeNodes StaticFilesManager
        try:
            from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
            static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                image_bytes, filename
            )
            print(f"[FLUX DEBUG] Image saved: {static_url}")
        except Exception as save_error:
            print(f"[FLUX DEBUG] StaticFilesManager failed: {save_error}")
            # Fallback to temp file
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                generated_image.save(tmp_file.name, format="PNG")
                static_url = f"file://{tmp_file.name}"
                print(f"[FLUX DEBUG] Fallback temp file: {static_url}")
        
        # Create output artifacts
        final_status = f"✅ Generated {width}x{height} image"
        if generation_info and isinstance(generation_info, dict):
            if 'actual_seed' in generation_info:
                final_status += f" (seed: {generation_info['actual_seed']})"
            if 'generation_time' in generation_info:
                final_status += f" in {generation_info['generation_time']:.1f}s"
        
        self.publish_update_to_parameter("status", final_status)
        print(f"[FLUX DEBUG] Final status: {final_status}")
        
        # Try to create ImageUrlArtifact for better UX
        try:
            from griptape.artifacts import ImageUrlArtifact
            image_artifact = ImageUrlArtifact(value=static_url)
            self.parameter_output_values["image"] = image_artifact
            print(f"[FLUX DEBUG] Created ImageUrlArtifact successfully")
        except Exception as artifact_error:
            print(f"[FLUX DEBUG] Artifact creation failed: {artifact_error}")
            # Fallback to simple string return
            self.parameter_output_values["image"] = static_url
            self.parameter_output_values["generation_info"] = str(generation_info)
            return static_url
            
        # Return generation info as string for the parameter
        self.parameter_output_values["generation_info"] = str(generation_info)
        
        print(f"[FLUX DEBUG] ===== GENERATION COMPLETE =====")
        print(f"[FLUX DEBUG] Image URL: {static_url}")
        print(f"[FLUX DEBUG] Generation info: {generation_info}")
        return static_url

def _create_error_image(self, error_msg: str, exception: Exception) -> str:
    return create_error_image(Path("/tmp"), error_msg, exception)