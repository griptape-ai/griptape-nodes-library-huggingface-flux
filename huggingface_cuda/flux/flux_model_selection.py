from typing import Any, List
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.traits.options import Options

# We reuse the model scan utility from flux_inference
try:
    from .flux_inference import FluxInference  # Local relative import
    print("[MODEL SELECTION DEBUG] Imported FluxInference via relative import")
except ImportError as e:
    print(f"[MODEL SELECTION DEBUG] Relative import failed: {e}")
    try:
        # Fallback to absolute import when node is loaded standalone
        from huggingface_cuda.flux.flux_inference import FluxInference
        print("[MODEL SELECTION DEBUG] Imported FluxInference via absolute import")
    except ImportError as e2:
        print(f"[MODEL SELECTION DEBUG] Absolute import failed: {e2}")
        try:
            import importlib.util, os, sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fi_path = os.path.join(current_dir, "flux_inference.py")
            print(f"[MODEL SELECTION DEBUG] Loading FluxInference via file path: {fi_path}")
            spec = importlib.util.spec_from_file_location("flux_inference_fallback", fi_path)
            fi_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fi_module)
            FluxInference = fi_module.FluxInference  # type: ignore
            print("[MODEL SELECTION DEBUG] Imported FluxInference via file loader fallback")
        except Exception as e3:
            print(f"[MODEL SELECTION DEBUG] File loader fallback failed: {e3}")
            raise

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
            tmp = FluxInference()
            models = tmp._scan_available_models()
            return models or ["auto"]
        except Exception:
            # Fallback to defaults
            return list(FluxInference.FLUX_MODELS.keys())

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
