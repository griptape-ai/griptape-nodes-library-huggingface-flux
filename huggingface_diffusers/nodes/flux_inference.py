from typing import Optional, Any
import torch
from io import BytesIO
import time
import logging
from diffusers import FluxPipeline
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape.artifacts import ImageArtifact, TextArtifact, ImageUrlArtifact
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
import re
import inspect
import json
import random


class FluxInference(ControlNode):
    _last_used_seed: int = 12345

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Diffusers"
        self.description = "Reference FLUX inference via Diffusers"
        self._logger = logging.getLogger(__name__)

        # Inputs/properties
        self.add_parameter(
            Parameter(
                name="model_config",
                input_types=["dict"],
                type="dict",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Model Config", "hide_property": True},
                tooltip="ModelConfig dict from selector (contains local_path)",
            )
        )
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Prompt text",
            )
        )
        self.add_parameter(
            Parameter(
                name="negative_prompt",
                input_types=["str"],
                type="str",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Negative prompt",
            )
        )
        self.add_parameter(
            Parameter(
                name="num_inference_steps",
                type="int",
                default_value=28,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Sampling steps",
            )
        )
        self.add_parameter(
            Parameter(
                name="guidance_scale",
                type="float",
                default_value=3.5,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="CFG guidance scale (dev)",
            )
        )
        self.add_parameter(
            Parameter(
                name="height",
                type="int",
                default_value=1024,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Image height",
            )
        )
        self.add_parameter(
            Parameter(
                name="width",
                type="int",
                default_value=1024,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Image width",
            )
        )
        self.add_parameter(
            Parameter(
                name="batch_size",
                type="int",
                default_value=1,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Number of images (1-4)",
                traits={Options(choices=[1, 2, 3, 4])},
            )
        )

        # Seed controls for downstream reproducibility (seed doubles as output for actual used seed)
        self.add_parameter(
            Parameter(
                name="seed",
                type="int",
                default_value=12345,
                allowed_modes={
                    ParameterMode.PROPERTY,
                    ParameterMode.INPUT,
                    ParameterMode.OUTPUT,
                },
                tooltip="Seed for reproducibility (outputs actual used seed)",
            )
        )
        self.add_parameter(
            Parameter(
                name="seed_control",
                type="str",
                default_value="randomize",
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Seed control mode",
                traits={
                    Options(choices=["fixed", "increment", "decrement", "randomize"])
                },
                ui_options={"display_name": "Seed Control"},
            )
        )

        # Grid list + positional outputs (reference pattern)
        self.add_parameter(
            Parameter(
                name="images",
                type="list",
                default_value=[],
                output_type="list[ImageUrlArtifact]",
                tooltip="Generated image artifacts (up to 4)",
                ui_options={"display": "grid", "columns": 2},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="image_1_1",
                type="ImageUrlArtifact",
                output_type="ImageUrlArtifact",
                tooltip="Image at grid position [1,1]",
                ui_options={"hide_property": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="image_1_2",
                type="ImageUrlArtifact",
                output_type="ImageUrlArtifact",
                tooltip="Image at grid position [1,2]",
                ui_options={"hide_property": True, "hide_when": {"batch_size": [1]}},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="image_2_1",
                type="ImageUrlArtifact",
                output_type="ImageUrlArtifact",
                tooltip="Image at grid position [2,1]",
                ui_options={"hide_property": True, "hide_when": {"batch_size": [1, 2]}},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="image_2_2",
                type="ImageUrlArtifact",
                output_type="ImageUrlArtifact",
                tooltip="Image at grid position [2,2]",
                ui_options={
                    "hide_property": True,
                    "hide_when": {"batch_size": [1, 2, 3]},
                },
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Performance report for DisplayJson node
        self.add_parameter(
            Parameter(
                name="performance_report",
                output_type="json",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Timing and settings per run",
            )
        )

        # Ensure initial visibility matches default batch size
        try:
            default_batch = int(self.get_parameter_value("batch_size") or 1)
            self._logger.info(
                f"[FluxInference] initial batch_size={default_batch}; applying visibility"
            )
            self._apply_positional_visibility(default_batch)
        except Exception:
            # Fallback to hiding all but first slot
            try:
                self._apply_positional_visibility(1)
            except Exception:
                pass

    def process(self) -> Any:
        yield lambda: self._run()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "batch_size":
            try:
                batch_val = int(value or 1)
                self._logger.info(
                    f"[FluxInference] batch_size changed -> {batch_val}; updating visibility"
                )
                self._apply_positional_visibility(batch_val)
            except Exception:
                pass
        return super().after_value_set(parameter, value)

    def _apply_positional_visibility(self, batch: int) -> None:
        # Show/hide positional outputs immediately (no run needed)
        layout = ["image_1_1", "image_1_2", "image_2_1", "image_2_2"]
        visible = [1, 2, 3, 4][: max(1, min(4, batch))]
        for idx, name in enumerate(layout, start=1):
            try:
                if idx in visible:
                    self.show_parameter_by_name(name)
                else:
                    self.hide_parameter_by_name(name)
            except Exception:
                pass

    def _run(self):
        try:
            # Clear previous outputs for a clean UI before starting inference
            for i in range(1, 5):
                slot = f"image_{1 + (i - 1) // 2}_{1 + (i - 1) % 2}"
                self.parameter_output_values[slot] = None
                try:
                    self.publish_update_to_parameter(slot, None)
                except Exception:
                    pass
            try:
                self.parameter_output_values["performance_report"] = {}
                self.publish_update_to_parameter("performance_report", {})
            except Exception:
                pass
            # Read model_config robustly (can arrive as dict, artifact, or JSON string)
            raw_cfg = self.get_parameter_value("model_config")
            cfg: dict = {}
            try:
                if isinstance(raw_cfg, dict):
                    cfg = raw_cfg
                elif hasattr(raw_cfg, "value"):
                    v = getattr(raw_cfg, "value")
                    if isinstance(v, dict):
                        cfg = v
                    elif isinstance(v, str):
                        cfg = json.loads(v)
                elif isinstance(raw_cfg, str):
                    cfg = json.loads(raw_cfg)
            except Exception:
                cfg = {}
            repo_id = cfg.get("model_id", "unknown")
            local_path = cfg.get("local_path")
            if not local_path:
                self._logger.error(
                    f"model_config missing local_path. cfg keys={list(cfg.keys()) if isinstance(cfg, dict) else type(cfg)}"
                )
            prompt = self.get_parameter_value("prompt") or "a photo of a spaceship"
            negative_prompt = self.get_parameter_value("negative_prompt")
            steps = int(self.get_parameter_value("num_inference_steps") or 28)
            guidance = float(self.get_parameter_value("guidance_scale") or 3.5)
            height = int(self.get_parameter_value("height") or 1024)
            width = int(self.get_parameter_value("width") or 1024)
            batch = max(1, min(4, int(self.get_parameter_value("batch_size") or 1)))

            # Seed control logic (fixed/increment/decrement/randomize)
            requested_seed = int(self.get_parameter_value("seed") or 12345)
            seed_mode = (
                self.get_parameter_value("seed_control") or "randomize"
            ).lower()
            if seed_mode == "fixed":
                actual_seed = requested_seed
            elif seed_mode == "increment":
                actual_seed = FluxInference._last_used_seed + 1
            elif seed_mode == "decrement":
                actual_seed = FluxInference._last_used_seed - 1
            else:
                actual_seed = random.randint(0, 2**32 - 1)
            actual_seed = max(0, min(actual_seed, 2**32 - 1))
            FluxInference._last_used_seed = actual_seed

            # Basic HF usage: prefer bfloat16 on CUDA to avoid fp16 underflow/NaNs
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            if not local_path:
                raise Exception(
                    "No local_path provided in model_config. This node never downloads models."
                )
            # Basic HF pattern: construct pipeline; then use CPU offload when CUDA is available
            # Delegate to shared runner to keep node lean
            # Robust import that works whether this library is loaded as a package or loose module
            try:
                from shared.inference_runner import run_flux_inference  # type: ignore
            except Exception:
                try:
                    from ..shared.inference_runner import run_flux_inference  # type: ignore
                except Exception:
                    # Final fallback: import by file path
                    import importlib.util, os, sys  # type: ignore

                    base_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), os.pardir)
                    )
                    runner_path = os.path.join(
                        base_dir, "shared", "inference_runner.py"
                    )
                    spec = importlib.util.spec_from_file_location(
                        "_hf_diffusers_inference_runner", runner_path
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = mod
                        spec.loader.exec_module(mod)
                        run_flux_inference = getattr(mod, "run_flux_inference")
                    else:
                        raise ImportError("Unable to load shared.inference_runner")
            # No CPU offload for this run to keep numerics stable; we'll re-introduce later if needed

            # Tokenizer length handled inside shared runner; keep node minimal

            prompts = [prompt] * batch
            negs = [negative_prompt] * batch if negative_prompt else None

            # Check what parameters the inference runner supports
            sig = inspect.signature(run_flux_inference)
            kwargs = {
                "local_path": local_path,
                "repo_id": repo_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "height": height,
                "width": width,
                "num_images_per_prompt": batch,
            }

            # Let inference runner use its default values for device_policy and quantization_mode

            images_png, timings = run_flux_inference(**kwargs)

            self._logger.info(
                f"load_pipe_s={timings['load_pipe_s']:.2f} infer_s={timings['infer_s']:.2f} encode_s={timings['encode_s']:.2f} total_s={timings['total_s']:.2f}"
            )
            # Emit JSON-like dict for DisplayJson
            self.parameter_output_values["performance_report"] = {
                "model_id": repo_id,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "seed_mode": seed_mode,
                "requested_seed": requested_seed,
                "actual_seed": actual_seed,
                "quantization_mode": timings.get("quantization_mode", "none"),
                **timings,
            }

            from uuid import uuid4

            t2 = time.perf_counter()
            # Save images and populate outputs
            images_list = []
            for idx in range(1, min(batch, len(images_png)) + 1):
                data = images_png[idx - 1]
                try:
                    model_short = (
                        (repo_id.split("/")[-1] if "/" in repo_id else repo_id)
                        .replace(".", "")
                        .replace("-", "_")
                        .lower()
                    )
                    filename = f"flux_{model_short}_seed_{actual_seed}_{idx}.png"
                    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
                    static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                        data, filename
                    )
                    art = ImageUrlArtifact(value=static_url, name=f"Image {idx}")
                except Exception:
                    art = ImageArtifact(
                        value=data, format="PNG", width=width, height=height
                    )
                images_list.append(art)
                # Map to positional outputs
                row = (idx - 1) // 2 + 1
                col = (idx - 1) % 2 + 1
                param_name = f"image_{row}_{col}"
                self.parameter_output_values[param_name] = art
                try:
                    self.publish_update_to_parameter(param_name, art)
                except Exception:
                    pass
            # Set grid list
            try:
                self.parameter_output_values["images"] = images_list
                self.publish_update_to_parameter("images", images_list)
            except Exception:
                pass
            t_save = time.perf_counter() - t2
            self._logger.info(f"save_s={t_save:.2f} total_s={timings['total_s']:.2f}")
            # Seed output on the same parameter: expose actual used seed
            try:
                self.parameter_output_values["seed"] = actual_seed
                self.publish_update_to_parameter("seed", actual_seed)
            except Exception:
                pass
        except Exception as e:
            self.parameter_output_values["image_1"] = None
            raise Exception(f"Diffusers inference error: {e}")
