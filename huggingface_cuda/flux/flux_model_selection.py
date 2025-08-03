from typing import Any, List
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.traits.options import Options

# Import the specific utilities we need instead of the heavy FluxInference class
try:
    from .flux_model_scanner import FluxModelScanner, FLUX_MODELS
except ImportError:
    # Fallback for standalone loading
    import importlib.util, os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load FluxModelScanner and FLUX_MODELS from the same file
    scanner_path = os.path.join(current_dir, "flux_model_scanner.py")
    spec = importlib.util.spec_from_file_location("flux_model_scanner", scanner_path)
    scanner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner_module)
    FluxModelScanner = scanner_module.FluxModelScanner
    FLUX_MODELS = scanner_module.FLUX_MODELS

class FluxModelSelection(ControlNode):
    """Light-weight node that lets user pick FLUX model and quantization options.
    Outputs validated selections so downstream inference node can load exactly
    one variant, avoiding mid-flow quantization switches.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Flux Model"
        self.description = (
            "Select FLUX model and quantization level. All components will be loaded on GPU."
        )

        # Discover cached models and encoder repos
        self._model_choices = self._discover_models()
        self._clip_choices, self._t5_choices = self._discover_encoders()

        with ParameterGroup(name="Model") as grp:
            self.add_parameter(
                Parameter(
                    name="model_id",
                    tooltip="Flux model to use",
                    type="str",
                    default_value=self._model_choices[0] if self._model_choices else "auto",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=self._model_choices)},
                    ui_options={"display_name": "Flux Model"},
                )
            )
            self.add_parameter(
                Parameter(
                    name="quantization",
                    tooltip="Quantization level (GPU only)",
                    type="str",
                    default_value="8-bit",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=["none", "4-bit", "8-bit"])},
                    ui_options={"display_name": "Quantization"},
                )
            )
                    # Encoders
            self.add_parameter(
                Parameter(
                    name="clip_id",
                    tooltip="CLIP text encoder repository (optional)",
                    type="str",
                    default_value=self._clip_choices[0] if self._clip_choices else "auto",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=self._clip_choices)},
                    ui_options={"display_name": "CLIP Encoder"},
                )
            )
            self.add_parameter(
                Parameter(
                    name="t5_id",
                    tooltip="T5 text encoder repository (optional)",
                    type="str",
                    default_value=self._t5_choices[0] if self._t5_choices else "auto",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=self._t5_choices)},
                    ui_options={"display_name": "T5 Encoder"},
                )
            )
        self.add_node_element(grp)

        # Status message (hidden output)
        self.add_parameter(
            Parameter(
                name="status",
                output_type="str",
                tooltip="Status updates",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"hide": True},
            )
        )

        # Single flux_config output (dict)
        self.add_parameter(
            Parameter(
                name="flux_config",
                output_type="dict",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Flux configuration dict with keys: model_id, quantization, clip_id, t5_id",
            )
        )

    # ---------------------------------------------------------------------
    def _discover_encoders(self) -> tuple[List[str], List[str]]:
        """Return (clip_repos, t5_repos) from HF cache"""
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            clip_ids, t5_ids = [], []
            for repo in cache.repos:
                lower = repo.repo_id.lower()
                if any(x in lower for x in ["clip", "clip-text", "clip_text"]):
                    clip_ids.append(repo.repo_id)
                if "t5" in lower and "encoder" in lower:
                    t5_ids.append(repo.repo_id)
            # UI select elements cannot have empty-string values.
            # Provide explicit placeholder 'auto' when nothing found.
            return clip_ids or ["auto"], t5_ids or ["auto"]
        except Exception:
            return ["auto"], ["auto"]

    # ---------------------------------------------------------------------
    def _discover_models(self) -> list[str]:
        try:
            scanner = FluxModelScanner()
            models = scanner.scan_available_models()
            return models or ["auto"]
        except Exception:
            # Fallback to defaults
            return list(FLUX_MODELS.keys())

    # ---------------------------------------------------------------------
    def process(self) -> None:
        self._process()

    def _process(self):
        model_id = self.get_parameter_value("model_id")
        quant = self.get_parameter_value("quantization")

        # Basic validation (quant option exists, model exists)
        if model_id not in self._model_choices:
            raise ValueError(f"Model {model_id} is not available on this system")
        if quant not in ["none", "4-bit", "8-bit"]:
            raise ValueError("Quantization must be none / 4-bit / 8-bit")

        self.publish_update_to_parameter("status", f"Selected {model_id} ({quant})")
        clip_id = self.get_parameter_value("clip_id")
        t5_id = self.get_parameter_value("t5_id")
        self.parameter_output_values["flux_config"] = {
            "model_id": model_id,
            "quantization": quant,
            "clip_id": clip_id,
            "t5_id": t5_id,
        }
