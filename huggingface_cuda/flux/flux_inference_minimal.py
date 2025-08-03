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

from .flux_config import FluxConfig
from .flux_cuda_helpers import PromptBuilder, ImageInputManager, create_error_image, SeedManager
from .flux_model_scanner import FluxModelScanner
from .flux_parameter_builder import FluxParameterBuilder
from .flux_result_processor import FluxResultProcessor

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

        # Initialize utility components
        self._model_scanner = FluxModelScanner()
        self._parameter_builder = FluxParameterBuilder(self)
        self._result_processor = FluxResultProcessor(self)

        # Initialize available models by scanning cache
        scan_start = time.time()
        print(f"[FLUX INIT] Scanning available models...")
        available_models = self._model_scanner.scan_available_models()
        scan_time = time.time() - scan_start
        print(f"[FLUX INIT] Model scanning completed in {scan_time:.2f}s -> Found {len(available_models)} models: {available_models}")
        
        # Build all UI parameters using the parameter builder
        print(f"[FLUX INIT] Building UI parameters...")
        param_start = time.time()
        self._parameter_builder.build_all_parameters(backend_name)
        param_time = time.time() - param_start
        print(f"[FLUX INIT] Parameters built in {param_time:.3f}s")
        
        # Log total initialization time
        total_init_time = time.time() - init_start
        print(f"[FLUX INIT] ✅ FluxInference initialization completed in {total_init_time:.2f}s")

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
        # This method can be enhanced to adjust parameters based on model selection
        pass

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
        
        # Process the result using the result processor
        static_url = self._result_processor.process_generation_result(
            backend_result, model_id, seed, width, height, guidance_scale, num_steps
        )
        
        return static_url


def _create_error_image(self, error_msg: str, exception: Exception) -> str:
    return create_error_image(Path("/tmp"), error_msg, exception)