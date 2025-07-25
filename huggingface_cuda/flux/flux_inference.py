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
    def generate_image(self, **kwargs) -> tuple[Any, dict]:
        """Generate image and return (pil_image, generation_info)"""
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
            backend = DiffusersFluxBackend()
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
    
    def generate_image(self, **kwargs) -> tuple[Any, dict]:
        """Delegate to actual backend, initializing if needed"""
        backend = self._ensure_backend()
        
        # DEBUG MODE: Just test the backend connection and quantization setup
        print(f"[FLUX DEBUG] 🎯 Backend ready: {type(backend).__name__}")
        print(f"[FLUX DEBUG] 🎯 Model requested: {kwargs.get('model_id', 'unknown')}")
        print(f"[FLUX DEBUG] 🎯 Quantization: {kwargs.get('quantization', 'none')}")
        
        # Test if quantization setup works
        if kwargs.get('quantization', 'none') in ['4-bit', '8-bit']:
            try:
                # Just test the quantization config creation without actual model loading
                quantization = kwargs.get('quantization', 'none')
                backend._test_quantization_setup(quantization)
                print(f"[FLUX DEBUG] ✅ Quantization setup test successful for {quantization}")
            except Exception as e:
                print(f"[FLUX DEBUG] ❌ Quantization setup test failed: {e}")
        
        print(f"[FLUX DEBUG] 🎯 DEBUG MODE: Skipping actual inference to avoid crashes")
        
        # Return dummy successful result
        return None, {"debug_mode": True, "status": "backend_connection_test_passed"}


class DiffusersFluxBackend(FluxBackend):
    """Diffusers-based backend for CUDA/CPU using shared backend"""
    
    def __init__(self):
        self._pipeline_cache: Dict[str, Any] = {}
        self._shared_backend = get_shared_backend()
    
    def is_available(self) -> bool:
        """Check if backend is available using pre-loaded modules"""
        if not self._shared_backend['available']:
            print(f"[FLUX BACKEND] Shared backend not available: {self._shared_backend.get('error', 'Unknown error')}")
            return False
        
        # Quick availability check using pre-loaded modules
        try:
            torch = self._shared_backend['torch']
            print(f"[FLUX BACKEND] Using pre-loaded PyTorch, CUDA available: {torch.cuda.is_available()}")
            return True
        except Exception as e:
            print(f"[FLUX BACKEND] Error checking availability: {e}")
            return False
    
    def get_name(self) -> str:
        """Get backend name using pre-loaded modules"""
        try:
            torch = self._shared_backend['torch']
            if torch.cuda.is_available():
                return "Diffusers (CUDA)"
            else:
                return "Diffusers (CPU)"
        except Exception:
            return "Diffusers"
    
    def validate_model_id(self, model_id: str) -> bool:
        return "FLUX.1-dev" in model_id or "FLUX.1-schnell" in model_id
    
    def _test_quantization_setup(self, quantization: str) -> bool:
        """Test quantization setup without loading models"""
        print(f"[FLUX DEBUG] Testing {quantization} quantization setup...")
        
        try:
            if not self._shared_backend['available']:
                raise ImportError(f"Shared backend not available: {self._shared_backend.get('error', 'Unknown error')}")
            
            BitsAndBytesConfig = self._shared_backend['BitsAndBytesConfig']
            torch = self._shared_backend['torch']
            
            print(f"[FLUX DEBUG] BitsAndBytesConfig available, testing {quantization} configuration...")
            
            if quantization == "4-bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                print(f"[FLUX DEBUG] ✅ Created 4-bit BitsAndBytesConfig")
            elif quantization == "8-bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    load_in_8bit_fp32_cpu_offload=True  # Enable CPU offload for insufficient GPU memory
                )
                print(f"[FLUX DEBUG] ✅ Created 8-bit BitsAndBytesConfig with CPU offload")
            else:
                raise ValueError(f"Unsupported quantization: {quantization}")
            
            print(f"[FLUX DEBUG] ✅ {quantization} quantization configuration test successful")
            return True
            
        except Exception as e:
            print(f"[FLUX DEBUG] ❌ {quantization} quantization test failed: {e}")
            raise
    
    def _get_pipeline(self, model_id: str, quantization: str = "none") -> Any:
        """Load diffusers pipeline with caching and optional quantization using shared backend"""
        cache_key = f"{model_id}_{quantization}"
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]
        
        # Use pre-loaded modules from shared backend
        if not self._shared_backend['available']:
            raise ImportError(f"Shared backend not available: {self._shared_backend.get('error', 'Unknown error')}")
        
        FluxPipeline = self._shared_backend['FluxPipeline']
        torch = self._shared_backend['torch']
        
        # Setup quantization config if needed using shared backend
        quantization_config = None
        bnb_config = None
        
        if quantization in ["4-bit", "8-bit"]:
            try:
                BitsAndBytesConfig = self._shared_backend['BitsAndBytesConfig']
                print(f"[FLUX DEBUG] BitsAndBytesConfig available, setting up {quantization} quantization...")
                
                if quantization == "4-bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    print(f"[FLUX DEBUG] ✅ Created 4-bit BitsAndBytesConfig")
                elif quantization == "8-bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        load_in_8bit_fp32_cpu_offload=True  # Enable CPU offload for insufficient GPU memory
                    )
                    print(f"[FLUX DEBUG] ✅ Created 8-bit BitsAndBytesConfig with CPU offload")
                
                # Try different approaches for quantization config
                quantization_config = None
                
                # Method 1: Try direct BitsAndBytesConfig (modern diffusers)
                try:
                    print(f"[FLUX DEBUG] 🔄 Trying direct BitsAndBytesConfig approach...")
                    quantization_config = bnb_config
                    print(f"[FLUX DEBUG] ✅ Using direct BitsAndBytesConfig for {quantization} quantization")
                except Exception as e1:
                    print(f"[FLUX DEBUG] ❌ Direct BitsAndBytesConfig failed: {e1}")
                    
                # Method 2: Try PipelineQuantizationConfig if direct approach failed
                if quantization_config is None:
                    try:
                        from diffusers import PipelineQuantizationConfig
                        quantization_config = PipelineQuantizationConfig(
                            quant_backend="bitsandbytes",
                            quant_kwargs=bnb_config.__dict__,
                            components_to_quantize=["transformer"]
                        )
                        print(f"[FLUX DEBUG] ✅ Using PipelineQuantizationConfig for {quantization} quantization")
                    except Exception as e2:
                        print(f"[FLUX DEBUG] ❌ PipelineQuantizationConfig failed: {e2}")
                        
                # Method 3: Disable quantization if all methods fail
                if quantization_config is None:
                    print(f"[FLUX DEBUG] ⚠️ All quantization methods failed, disabling quantization")
                    bnb_config = None
                    
            except Exception as e:
                print(f"[FLUX ERROR] ❌ Quantization setup failed: {e}")
                print(f"[FLUX ERROR] ❌ Falling back to full precision - this will use ~25GB+ memory!")
                print(f"[FLUX ERROR] 💡 To fix: pip install bitsandbytes>=0.41.0")
                quantization_config = None
                bnb_config = None
        
        # Load pipeline - try different quantization approaches
        pipeline = None
        
        if quantization_config is not None:
            # Method 1: Try with component-level quantization
            try:
                print(f"[FLUX LOADING] 🔄 Loading {model_id} with component-level quantization ({quantization})...")
                print(f"[FLUX LOADING] 💾 This should use much less memory than full precision")
                
                # Load transformer with quantization
                from diffusers import FluxTransformer2DModel
                
                # Set up loading parameters based on quantization type
                loading_kwargs = {
                    "subfolder": "transformer",
                    "quantization_config": quantization_config,
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto"
                }
                
                if quantization == "8-bit":
                    # For 8-bit with CPU offload, add memory management parameters
                    loading_kwargs.update({
                        "low_cpu_mem_usage": True,
                        "max_memory": {0: "15GB", "cpu": "20GB"}  # Conservative GPU limit to trigger offload
                    })
                    print(f"[FLUX LOADING] Using 8-bit with memory limits to enable CPU offload")
                else:
                    print(f"[FLUX LOADING] Using standard auto device map for {quantization}")
                
                transformer = FluxTransformer2DModel.from_pretrained(model_id, **loading_kwargs)
                
                # Load pipeline with quantized transformer
                pipeline = FluxPipeline.from_pretrained(
                    model_id,
                    transformer=transformer,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                )
                
                # Ensure all pipeline components are on the same device
                if torch.cuda.is_available():
                    try:
                        pipeline = pipeline.to("cuda")
                        print(f"[FLUX LOADING] 🔄 Moved pipeline to CUDA device")
                    except Exception as e:
                        print(f"[FLUX LOADING] ⚠️ Could not move full pipeline to CUDA: {e}")
                        # Try to move individual components that aren't quantized
                        try:
                            if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                                pipeline.text_encoder = pipeline.text_encoder.to("cuda")
                            if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
                                pipeline.text_encoder_2 = pipeline.text_encoder_2.to("cuda")
                            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                                pipeline.vae = pipeline.vae.to("cuda")
                            print(f"[FLUX LOADING] 🔄 Moved individual components to CUDA")
                        except Exception as e2:
                            print(f"[FLUX LOADING] ⚠️ Device placement warning: {e2}")
                
                print(f"[FLUX LOADING] ✅ Successfully loaded with component-level quantization")
                
            except Exception as e1:
                print(f"[FLUX LOADING] ❌ Component-level quantization failed: {e1}")
                
                # Method 2: Try pipeline-level quantization as fallback
                try:
                    print(f"[FLUX LOADING] 🔄 Trying pipeline-level quantization...")
                    pipeline = FluxPipeline.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        device_map="auto"
                    )
                    print(f"[FLUX LOADING] ✅ Successfully loaded with pipeline-level quantization")
                except Exception as e2:
                    print(f"[FLUX LOADING] ❌ Pipeline-level quantization also failed: {e2}")
                    pipeline = None
        
        if pipeline is None:
            # Fallback to full precision
            print(f"[FLUX LOADING] ⚠️ Loading {model_id} without quantization (FULL PRECISION)")
            print(f"[FLUX LOADING] ⚠️ This will use ~25GB+ memory and may cause OOM!")
            pipeline = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            
            # Move to appropriate device for non-quantized models
            if torch.cuda.is_available():
                try:
                    pipeline = pipeline.to("cuda")
                    print(f"[FLUX LOADING] 🔄 Moved full-precision pipeline to CUDA")
                except Exception as e:
                    print(f"[FLUX LOADING] ⚠️ Could not move pipeline to CUDA: {e}")
        
        self._pipeline_cache[cache_key] = pipeline
        return pipeline
    
    def generate_image(self, **kwargs) -> tuple[Any, dict]:
        # Use pre-loaded torch from shared backend
        torch = self._shared_backend['torch']
        
        model_id = kwargs['model_id']
        prompt = kwargs['prompt']
        width = kwargs['width']
        height = kwargs['height']
        steps = kwargs['steps']
        guidance = kwargs['guidance']
        seed = kwargs['seed']
        max_sequence_length = kwargs.get('max_sequence_length', 512)
        quantization = kwargs.get('quantization', 'none')
        
        pipeline = self._get_pipeline(model_id, quantization)
        
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
            "seed": seed,
            "quantization": quantization
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
        import time
        init_start = time.time()
        print(f"[FLUX INIT] Starting FluxInference initialization...")
        
        super().__init__(**kwargs)
        self.category = "Flux CUDA"
        
        # Use pre-loaded shared backend (fast!)
        backend_start = time.time()
        print(f"[FLUX INIT] Using shared CUDA backend...")
        self._backend = self._detect_optimal_backend()
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
            
            self.add_parameter(
                Parameter(
                    name="quantization",
                    tooltip="Model quantization to reduce memory usage. Lower bit = less memory but potentially lower quality.",
                    type="str",
                    default_value="none", 
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=["none", "4-bit", "8-bit"])},
                    ui_options={"display_name": "Quantization"}
                )
            )
        
        gen_group.ui_options = {"collapsed": False}
        self.add_node_element(gen_group)

        # Note: Using unified parameters like MLX version instead of model-specific groups

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

    def _detect_optimal_backend(self) -> FluxBackend:
        """Return best available backend, using deferred loading if needed"""
        import time
        print(f"[FLUX INIT] Creating backend...")
        backend_start = time.time()
        
        # Try to create the full backend first
        try:
            diffusers_backend = DiffusersFluxBackend()
            backend_time = time.time() - backend_start
            print(f"[FLUX INIT] Backend instance created in {backend_time:.2f}s")
            
            print(f"[FLUX INIT] Checking backend availability...")
            avail_start = time.time()
            is_available = diffusers_backend.is_available()
            avail_time = time.time() - avail_start
            print(f"[FLUX INIT] Availability check completed in {avail_time:.2f}s -> {is_available}")
            
            if is_available:
                print(f"[FLUX INIT] Using Diffusers backend for CUDA/CPU")
                return diffusers_backend
            else:
                print(f"[FLUX INIT] Dependencies not ready, using deferred backend")
                return DeferredFluxBackend()
                
        except Exception as e:
            print(f"[FLUX INIT] Backend creation failed ({e}), using deferred backend")
            return DeferredFluxBackend()

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
                        available.append(repo_id)
                        print(f"[FLUX SCAN] Found base FLUX model: {repo_id}")
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

    def after_value_set(self, parameter: Parameter, value: Any, modified_parameters_set: set[str] | None = None) -> None:
        """Handle parameter changes, especially model selection."""
        if parameter.name == "model":
            self._update_guidance_scale_for_model(value)

    def _update_guidance_scale_for_model(self, selected_model: str) -> None:
        """Update guidance scale based on model capabilities."""
        model_config = self.FLUX_MODELS.get(selected_model, {})
        
        if not model_config.get("supports_guidance", True):
            # For models that don't support guidance (like Schnell), set to 1.0
            self.set_parameter_value("guidance_scale", 1.0)
            print(f"[FLUX CONFIG] Updated guidance scale to 1.0 for {selected_model}")

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
                quantization = self.get_parameter_value("quantization")
                
                # Validate inputs - Set safe defaults before any validation failures
                self.parameter_output_values["image"] = None
                self.parameter_output_values["generation_info"] = "{}"
                self.parameter_output_values["status"] = ""
                
                if not model_id:
                    self.publish_update_to_parameter("status", "❌ No Flux model selected")
                    raise ValueError("No Flux model selected. Please select a model.")
                if not prompt or not prompt.strip():
                    self.publish_update_to_parameter("status", "❌ Prompt is required")
                    raise ValueError("Prompt is required for image generation.")
                
                # Validate model compatibility with backend
                if not self._backend.validate_model_id(model_id):
                    self.publish_update_to_parameter("status", f"❌ Model {model_id} not supported")
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
                
                # Use the steps parameter directly (like MLX version)
                steps = int(self.get_parameter_value("steps"))
                num_steps = steps
                max_sequence_length = 512
                
                # Override guidance for schnell model (which doesn't support guidance)
                if "FLUX.1-schnell" in model_id:
                    guidance_scale = 1.0  # Force correct guidance for schnell
                
                self.publish_update_to_parameter("status", f"🚀 Generating with {self._backend.get_name()} backend...")
                self.publish_update_to_parameter("status", f"📝 Prompt: '{enhanced_prompt[:50]}{'...' if len(enhanced_prompt) > 50 else ''}'")
                self.publish_update_to_parameter("status", f"⚙️ Settings: {width}x{height}, {num_steps} steps, guidance={guidance_scale}, quantization={quantization}")
                print(f"[FLUX DEBUG] Backend: {self._backend.get_name()}")
                print(f"[FLUX DEBUG] Model: {model_id}")
                print(f"[FLUX DEBUG] Enhanced prompt: '{enhanced_prompt}'")
                print(f"[FLUX DEBUG] Quantization: {quantization}")
                
                # Show quantization status
                if quantization == "none":
                    self.publish_update_to_parameter("status", f"🔧 Using full precision (no quantization)")
                elif quantization in ["4-bit", "8-bit"]:
                    self.publish_update_to_parameter("status", f"⚡ Using {quantization} quantization for memory efficiency")
                
                # Generate using backend
                generated_image, generation_info = self._backend.generate_image(
                    model_id=model_id,
                    prompt=enhanced_prompt,
                    width=width,
                    height=height,
                    steps=num_steps,
                    guidance=guidance_scale,
                    seed=seed,
                    max_sequence_length=max_sequence_length,
                    quantization=quantization
                )
                
                # Validate generated image
                if generated_image is None:
                    raise ValueError("Backend returned None image")
                
                # Check for black frames using numpy
                try:
                    # Try to get numpy from backend
                    np = self._backend._shared_backend['numpy']
                except (AttributeError, KeyError):
                    # Fallback to direct import
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