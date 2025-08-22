import os
from dataclasses import dataclass
from typing import List, Dict, Any

from griptape.artifacts import TextArtifact
import logging
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.traits.options import Options

from huggingface_hub import scan_cache_dir

# Try package-relative imports first; fall back to file import when nodes are loaded as loose modules
try:
    from ..shared.model_config import ModelConfig  # type: ignore
    from ..shared.hf_cache_scanner import HFCacheScanner  # type: ignore
except Exception:
    import importlib.util, pathlib

    base = pathlib.Path(__file__).resolve().parents[1] / "shared"
    for name in ("model_config", "hf_cache_scanner"):
        spec = importlib.util.spec_from_file_location(name, str(base / f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
        globals()[name] = mod
    ModelConfig = globals()["model_config"].ModelConfig
    HFCacheScanner = globals()["hf_cache_scanner"].HFCacheScanner


@dataclass
class FluxModelInfo:
    repo_id: str
    variant: str  # dev/schnell/etc


class FluxModelSelectorNode(ControlNode):
    """Scan HF cache for Flux repos and output a selected repo_id.
    UI exposes a dropdown populated from the cache scan (falls back to canonical IDs).
    Outputs: repo_id (str), variant (str)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Diffusers"
        self.description = "Discover Flux repos from Hugging Face cache and pick one"

        transformer_choices, clip_choices, t5_choices, model_mappings = (
            self._discover_choices()
        )
        # Keep mapping and expose choice lists like bkup
        self._model_mappings: Dict[str, str] = model_mappings
        self._transformer_choices = transformer_choices
        self._clip_choices = clip_choices
        self._t5_choices = t5_choices
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            f"Selector init: transformers={len(transformer_choices)}, clip={len(clip_choices)}, t5={len(t5_choices)}"
        )

        with ParameterGroup(name="Model") as grp:
            self.add_parameter(
                Parameter(
                    name="flux_transformer",
                    tooltip="FLUX transformer model to use (local cache only)",
                    type="str",
                    default_value=(
                        self._transformer_choices[0]
                        if self._transformer_choices
                        else ""
                    ),
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=self._transformer_choices)},
                    ui_options={"display_name": "FLUX Transformer"},
                )
            )

            self.add_parameter(
                Parameter(
                    name="clip_encoder",
                    tooltip="CLIP text encoder to use",
                    type="str",
                    default_value=(self._clip_choices[0] if self._clip_choices else ""),
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=self._clip_choices)},
                    ui_options={"display_name": "CLIP Encoder"},
                )
            )

            self.add_parameter(
                Parameter(
                    name="t5_encoder",
                    tooltip="T5 text encoder to use",
                    type="str",
                    default_value=(self._t5_choices[0] if self._t5_choices else ""),
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=self._t5_choices)},
                    ui_options={"display_name": "T5 Encoder"},
                )
            )

        self.add_node_element(grp)

        # Outputs
        self.add_parameter(
            Parameter(
                name="model_config",
                output_type="dict",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Serialized ModelConfig for inference node",
                ui_options={"display_name": "Model Config", "hide_property": True},
            )
        )

    def _discover_choices(
        self,
    ) -> tuple[List[str], List[str], List[str], Dict[str, str]]:
        # Prefer our scanner (no network) and build display->path map like bkup
        flux: List[str] = []
        clips: List[str] = []
        t5s: List[str] = []
        path_map: Dict[str, str] = {}
        try:
            scanner = HFCacheScanner()
            found = scanner.scan_flux_models()
            for m in found.get("transformers", []):
                display = m["display_name"]
                flux.append(display)
                path_map[display] = str(m["path"])
            for m in found.get("clip_encoders", []):
                display = m["display_name"]
                clips.append(display)
                path_map[display] = str(m["path"])
            for m in found.get("t5_encoders", []):
                display = m["display_name"]
                t5s.append(display)
                path_map[display] = str(m["path"])
        except Exception:
            pass
        return flux, clips, t5s, path_map

    def process(self) -> Any:
        yield lambda: self._run()

    def _run(self):
        transformer_display = self.get_parameter_value("flux_transformer")
        clip_display = self.get_parameter_value("clip_encoder")
        t5_display = self.get_parameter_value("t5_encoder")
        self._logger.warning(
            f"Selected: transformer='{transformer_display}', clip='{clip_display}', t5='{t5_display}'"
        )
        # Strict local-only: require resolved paths from cache scan; never fall back to repo IDs
        if transformer_display not in self._model_mappings:
            raise ValueError(
                "No local snapshot found for selected FLUX transformer. Download the model first using HF CLI or a Griptape HF node."
            )
        if clip_display not in self._model_mappings:
            raise ValueError(
                "No local snapshot found for selected CLIP encoder. Download it first."
            )
        if t5_display not in self._model_mappings:
            raise ValueError(
                "No local snapshot found for selected T5 encoder. Download it first."
            )
        transformer_path = self._model_mappings[transformer_display]
        clip_path = self._model_mappings[clip_display]
        t5_path = self._model_mappings[t5_display]
        self._logger.warning(
            f"Path map: transformer='{transformer_path}', clip='{clip_path}', t5='{t5_path}'"
        )
        rid = transformer_display
        variant = (
            "dev"
            if "dev" in rid.lower()
            else ("schnell" if "schnell" in rid.lower() else "unknown")
        )
        cfg = ModelConfig(
            model_id=rid,
            clip_id=clip_path,
            t5_id=t5_path,
            variant=variant,
            local_path=transformer_path,
        )
        self.parameter_output_values["model_config"] = cfg.to_dict()
        return TextArtifact(f"Selected {rid} ({variant})")
