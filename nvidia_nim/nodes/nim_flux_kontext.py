from typing import Any, Dict, List
import os
import json
import base64
import time
import re
import tempfile
import requests
from urllib.parse import urlparse
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options
from griptape.artifacts import TextArtifact, ImageUrlArtifact
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class NIMFluxKontext(ControlNode):
    _last_used_seed: int = 12345

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "NIM"
        self.description = (
            "Flux img2img/control variants (init/control images) with batching"
        )

        default_base = os.getenv("NIM_BASE_URL", "http://localhost:8000")

        # Inputs/properties
        self.add_parameter(
            Parameter(
                name="service_config",
                input_types=["dict"],
                type="dict",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Service Config", "hide_property": True},
                tooltip="Config from container (base_url, route, defaults).",
            )
        )
        self.add_parameter(
            Parameter(
                name="base_url",
                type="str",
                default_value=default_base,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Base URL of the NIM service.",
            )
        )
        self.add_parameter(
            Parameter(
                name="route",
                type="str",
                default_value="/v1/infer",
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Inference route (e.g., /v1/infer).",
            )
        )

        self.add_parameter(
            Parameter(
                name="mode",
                type="str",
                default_value="base",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=["base", "canny", "depth"])},
                tooltip="Flux variant/mode for control.",
            )
        )
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Text prompt.",
            )
        )
        self.add_parameter(
            Parameter(
                name="strength",
                type="float",
                default_value=0.7,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Img2img strength/denoise.",
            )
        )
        self.add_parameter(
            Parameter(
                name="width",
                type="int",
                default_value=1024,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Image width.",
            )
        )
        self.add_parameter(
            Parameter(
                name="height",
                type="int",
                default_value=1024,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Image height.",
            )
        )
        self.add_parameter(
            Parameter(
                name="steps",
                type="int",
                default_value=30,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Inference steps.",
            )
        )
        self.add_parameter(
            Parameter(
                name="cfg_scale",
                type="float",
                default_value=3.5,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="CFG guidance scale.",
            )
        )
        self.add_parameter(
            Parameter(
                name="samples",
                type="int",
                default_value=1,
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=[1, 2, 3, 4])},
                tooltip="Number of images to generate (1-4).",
            )
        )
        # Seed controls (move up for consistency)
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
                tooltip="Seed for reproducibility (outputs actual used seed).",
            )
        )
        self.add_parameter(
            Parameter(
                name="seed_control",
                type="str",
                default_value="randomize",
                allowed_modes={ParameterMode.PROPERTY},
                traits={
                    Options(choices=["fixed", "increment", "decrement", "randomize"])
                },
                tooltip="Seed control mode.",
            )
        )

        # Image inputs
        self.add_parameter(
            Parameter(
                name="init_image",
                input_types=["ImageUrlArtifact"],
                type="ImageUrlArtifact",
                allowed_modes={ParameterMode.INPUT},
                tooltip="Initial image for img2img (optional).",
            )
        )
        self.add_parameter(
            Parameter(
                name="control_image",
                input_types=["ImageUrlArtifact"],
                type="ImageUrlArtifact",
                allowed_modes={ParameterMode.INPUT},
                tooltip="Control image for canny/depth (optional).",
            )
        )

        # Outputs: group + positional images
        from griptape_nodes.exe_types.core_types import ParameterGroup

        with ParameterGroup(name="Images") as images_group:
            self.add_parameter(
                Parameter(
                    name="images",
                    type="list",
                    default_value=[],
                    output_type="list[ImageUrlArtifact]",
                    allowed_modes={ParameterMode.OUTPUT},
                    ui_options={"display": "grid", "columns": 2},
                    tooltip="All generated images (grid).",
                )
            )
            self.add_parameter(
                Parameter(
                    name="image_1_1",
                    type="ImageUrlArtifact",
                    output_type="ImageUrlArtifact",
                    allowed_modes={ParameterMode.OUTPUT},
                    ui_options={"hide_property": True},
                    tooltip="Image [1,1].",
                )
            )
            self.add_parameter(
                Parameter(
                    name="image_1_2",
                    type="ImageUrlArtifact",
                    output_type="ImageUrlArtifact",
                    allowed_modes={ParameterMode.OUTPUT},
                    ui_options={"hide_property": True, "hide_when": {"samples": [1]}},
                    tooltip="Image [1,2].",
                )
            )
            self.add_parameter(
                Parameter(
                    name="image_2_1",
                    type="ImageUrlArtifact",
                    output_type="ImageUrlArtifact",
                    allowed_modes={ParameterMode.OUTPUT},
                    ui_options={
                        "hide_property": True,
                        "hide_when": {"samples": [1, 2]},
                    },
                    tooltip="Image [2,1].",
                )
            )
            self.add_parameter(
                Parameter(
                    name="image_2_2",
                    type="ImageUrlArtifact",
                    output_type="ImageUrlArtifact",
                    allowed_modes={ParameterMode.OUTPUT},
                    ui_options={
                        "hide_property": True,
                        "hide_when": {"samples": [1, 2, 3]},
                    },
                    tooltip="Image [2,2].",
                )
            )
        self.add_node_element(images_group)
        self.add_parameter(
            Parameter(
                name="response",
                output_type="dict",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Response", "hide_property": True},
                tooltip="Raw JSON response.",
            )
        )
        self.add_parameter(
            Parameter(
                name="performance",
                output_type="json",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"hide_property": True},
                tooltip="Timing and settings per run.",
            )
        )

        # Ensure initial visibility matches default samples
        try:
            default_samples = int(self.get_parameter_value("samples") or 1)
            self._apply_positional_visibility(default_samples)
        except Exception:
            pass

    def process(self) -> Any:
        yield lambda: self._run()

    def _read_image_b64(self, artifact: Any) -> str | None:
        try:
            if not artifact:
                return None
            uri = (
                getattr(artifact, "value", None)
                or getattr(artifact, "uri", None)
                or getattr(artifact, "url", None)
            )
            if not isinstance(uri, str):
                return None
            p = urlparse(uri)
            # Support StaticFilesManager HTTP URLs and local files
            if p.scheme in ("http", "https"):
                resp = requests.get(uri, timeout=30)
                resp.raise_for_status()
                return base64.b64encode(resp.content).decode("utf-8")
            # file:// or direct path
            path = uri
            if p.scheme == "file":
                path = p.path or uri
            # Windows file://C:/... normalization
            if path.startswith("file://"):
                path = path[len("file://") :]
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    def _run(self):
        svc = self.get_parameter_value("service_config") or {}
        base = (
            self.get_parameter_value("base_url")
            or svc.get("base_url")
            or "http://localhost:8000"
        ).rstrip("/")
        route = (
            self.get_parameter_value("route") or svc.get("route") or "/v1/infer"
        ).lstrip("/")
        url = base + "/" + route

        prompt = self.get_parameter_value("prompt") or "A cozy room"
        mode = (
            self.get_parameter_value("mode")
            or (
                svc.get("defaults", {}).get("mode") if isinstance(svc, dict) else "base"
            )
            or "base"
        )
        width = int(self.get_parameter_value("width") or 1024)
        height = int(self.get_parameter_value("height") or 1024)
        steps = int(self.get_parameter_value("steps") or 30)
        cfg_scale = float(self.get_parameter_value("cfg_scale") or 3.5)
        samples = max(1, min(4, int(self.get_parameter_value("samples") or 1)))
        strength = float(self.get_parameter_value("strength") or 0.7)

        # Seed control
        try:
            requested_seed = int(self.get_parameter_value("seed") or 12345)
        except Exception:
            requested_seed = 12345
        seed_mode = (self.get_parameter_value("seed_control") or "randomize").lower()
        if seed_mode == "fixed":
            actual_seed = requested_seed
        elif seed_mode == "increment":
            actual_seed = NIMFluxKontext._last_used_seed + 1
        elif seed_mode == "decrement":
            actual_seed = NIMFluxKontext._last_used_seed - 1
        else:
            import random

            actual_seed = random.randint(0, 2**32 - 1)
        actual_seed = max(0, min(actual_seed, 2**32 - 1))
        NIMFluxKontext._last_used_seed = actual_seed
        self.parameter_output_values["seed"] = actual_seed

        init_image = self.get_parameter_value("init_image")
        control_image = self.get_parameter_value("control_image")
        init_b64 = self._read_image_b64(init_image)
        control_b64 = self._read_image_b64(control_image)

        # Build minimal payload per Kontext API: prompt, image (data URL), seed, steps
        body: Dict[str, Any] = {
            "prompt": prompt,
            "seed": int(actual_seed),
            "steps": steps,
        }
        # Build data URL for main image (prefer jpeg/png based on init artifact filename)
        if init_b64:
            # naive mime detection from filename
            mime = "image/jpeg"
            try:
                src = getattr(init_image, "value", "") or ""
                if isinstance(src, str) and src.lower().endswith(".png"):
                    mime = "image/png"
            except Exception:
                pass
            body["image"] = f"data:{mime};base64,{init_b64}"
        # Optional control image for canny/depth variants – send as data URL if present
        if control_b64:
            ctrl_mime = "image/png"
            try:
                src = getattr(control_image, "value", "") or ""
                if isinstance(src, str) and (
                    src.lower().endswith(".jpg") or src.lower().endswith(".jpeg")
                ):
                    ctrl_mime = "image/jpeg"
            except Exception:
                pass
            body["control_image"] = f"data:{ctrl_mime};base64,{control_b64}"
            # Include mode only when control image present
            body["mode"] = mode

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        resp = requests.post(url, headers=headers, json=body, timeout=600)
        try:
            js = resp.json()
        except Exception:
            js = {"status": resp.status_code, "text": resp.text}
        self.parameter_output_values["response"] = js

        images: List[ImageUrlArtifact] = []
        try:
            arts = js.get("artifacts") or []
            for idx, art in enumerate(arts[:4], start=1):
                if not isinstance(art, dict):
                    continue
                b64 = art.get("base64")
                if not isinstance(b64, str) or not b64:
                    continue
                mime = (art.get("mime_type") or art.get("mime") or "").lower()
                ext = ".jpg"
                if "png" in mime:
                    ext = ".png"
                elif "webp" in mime:
                    ext = ".webp"
                data = base64.b64decode(b64)

                route_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", route).strip("_")
                filename = f"nim_kontext_{route_slug}_seed_{actual_seed}_{idx}_{int(time.time() * 1000)}{ext}"
                try:
                    static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                        data, filename
                    )
                except Exception:
                    fd, path = tempfile.mkstemp(suffix=ext)
                    with os.fdopen(fd, "wb") as f:
                        f.write(data)
                    static_url = f"file://{path}"
                try:
                    img = ImageUrlArtifact(value=static_url, name=f"Image {idx}")
                except Exception:
                    img = ImageUrlArtifact(static_url)
                images.append(img)
        except Exception:
            pass

        try:
            self.parameter_output_values["images"] = images
        except Exception:
            pass
        layout = ["image_1_1", "image_1_2", "image_2_1", "image_2_2"]
        for i, name in enumerate(layout):
            val = images[i] if i < len(images) else None
            self.parameter_output_values[name] = val

        return TextArtifact(f"HTTP {resp.status_code}")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "samples":
            try:
                val = int(value or 1)
                self._apply_positional_visibility(val)
            except Exception:
                pass
        return super().after_value_set(parameter, value)

    def _apply_positional_visibility(self, samples: int) -> None:
        layout = ["image_1_1", "image_1_2", "image_2_1", "image_2_2"]
        visible = [1, 2, 3, 4][: max(1, min(4, samples))]
        for idx, name in enumerate(layout, start=1):
            try:
                if idx in visible:
                    self.show_parameter_by_name(name)
                else:
                    self.hide_parameter_by_name(name)
            except Exception:
                pass
