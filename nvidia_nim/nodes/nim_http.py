from typing import Any, Dict
import json
import os
import base64
import tempfile
import time
import re
import requests
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options
from griptape.artifacts import TextArtifact, ImageUrlArtifact
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

class NIMHTTPInference(ControlNode):
    _last_used_seed: int = 12345
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "NIM"
        self.description = "Call a NIM HTTP endpoint (OpenAI-compatible or model-specific)"

        default_base = os.getenv("NIM_BASE_URL", "http://localhost:8000")
        default_key = os.getenv("NIM_API_KEY", "")

        self.add_parameter(Parameter(name="service_config", input_types=["dict"], type="dict", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}, tooltip="Config dict from container node (base_url, route, defaults)."))
        self.add_parameter(Parameter(name="base_url", type="str", default_value=default_base, allowed_modes={ParameterMode.PROPERTY}, tooltip="Base URL of the NIM service."))
        self.add_parameter(Parameter(name="route", type="str", default_value="/v1/infer", allowed_modes={ParameterMode.PROPERTY}, tooltip="API route (e.g., /v1/infer or /v1/images/generations)."))
        self.add_parameter(Parameter(name="method", type="str", default_value="POST", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["GET", "POST"])}, tooltip="HTTP method."))
        self.add_parameter(Parameter(name="model", type="str", default_value="black-forest-labs/flux_1-dev", allowed_modes={ParameterMode.PROPERTY}, tooltip="Model identifier (if API expects it)."))
        self.add_parameter(Parameter(name="prompt", input_types=["str"], type="str", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}, tooltip="Prompt text (when not using raw JSON)."))
        self.add_parameter(Parameter(name="json_payload", input_types=["str"], type="str", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}, tooltip="Optional raw JSON payload to send instead of prompt/model."))
        self.add_parameter(Parameter(name="api_key", type="str", default_value=default_key, allowed_modes={ParameterMode.PROPERTY}, ui_options={"display_name": "API Key", "secret": True}, tooltip="Bearer token for NIM (if required)."))

        # Batch + seed controls (mirror Diffusers node ergonomics)
        self.add_parameter(Parameter(name="samples", type="int", default_value=1, allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=[1, 2, 3, 4])}, tooltip="Number of images to generate (1-4)"))
        self.add_parameter(Parameter(name="seed", type="int", default_value=12345, allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT, ParameterMode.OUTPUT}, tooltip="Seed for reproducibility (outputs actual used seed)"))
        self.add_parameter(Parameter(name="seed_control", type="str", default_value="randomize", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["fixed", "increment", "decrement", "randomize"])}, tooltip="Seed control mode"))

        self.add_parameter(Parameter(name="response", output_type="dict", allowed_modes={ParameterMode.OUTPUT}, ui_options={"display_name": "Response", "hide_property": True}, tooltip="Raw JSON response."))
        # Primary first image for quick wiring
        self.add_parameter(Parameter(name="image", output_type="ImageUrlArtifact", allowed_modes={ParameterMode.OUTPUT}, tooltip="First generated image as file URL."))

        # Grid outputs to mirror FluxInference UX
        self.add_parameter(Parameter(
            name="images",
            type="list",
            default_value=[],
            output_type="list[ImageUrlArtifact]",
            tooltip="Generated images (up to 4)",
            ui_options={"display": "grid", "columns": 2},
            allowed_modes={ParameterMode.OUTPUT},
        ))
        self.add_parameter(Parameter(
            name="image_1_1",
            type="ImageUrlArtifact",
            output_type="ImageUrlArtifact",
            tooltip="Image at grid position [1,1]",
            ui_options={"hide_property": True},
            allowed_modes={ParameterMode.OUTPUT},
        ))
        self.add_parameter(Parameter(
            name="image_1_2",
            type="ImageUrlArtifact",
            output_type="ImageUrlArtifact",
            tooltip="Image at grid position [1,2]",
            ui_options={"hide_property": True, "hide_when": {"samples": [1]}},
            allowed_modes={ParameterMode.OUTPUT},
        ))
        self.add_parameter(Parameter(
            name="image_2_1",
            type="ImageUrlArtifact",
            output_type="ImageUrlArtifact",
            tooltip="Image at grid position [2,1]",
            ui_options={"hide_property": True, "hide_when": {"samples": [1, 2]}},
            allowed_modes={ParameterMode.OUTPUT},
        ))
        self.add_parameter(Parameter(
            name="image_2_2",
            type="ImageUrlArtifact",
            output_type="ImageUrlArtifact",
            tooltip="Image at grid position [2,2]",
            ui_options={"hide_property": True, "hide_when": {"samples": [1, 2, 3]}},
            allowed_modes={ParameterMode.OUTPUT},
        ))

    def process(self) -> Any:
        yield lambda: self._run()

    def _run(self):
        # Merge service_config with explicit parameter precedence
        svc = self.get_parameter_value("service_config") or {}
        base_param = self.get_parameter_value("base_url")
        route_param = self.get_parameter_value("route")
        base = base_param or (svc.get("base_url") if isinstance(svc, dict) else None) or "http://localhost:8000"
        route = route_param or (svc.get("route") if isinstance(svc, dict) else None) or "/v1/infer"
        method = (self.get_parameter_value("method") or "POST").upper()
        prompt = self.get_parameter_value("prompt") or "A simple coffee shop interior"
        raw = self.get_parameter_value("json_payload")
        api_key = self.get_parameter_value("api_key") or ""
        model = self.get_parameter_value("model") or "black-forest-labs/flux_1-dev"
        defaults = svc.get("defaults") if isinstance(svc, dict) else None

        # Seed control (fixed/increment/decrement/randomize)
        try:
            requested_seed = int(self.get_parameter_value("seed") or 12345)
        except Exception:
            requested_seed = 12345
        seed_mode = (self.get_parameter_value("seed_control") or "randomize").lower()
        if seed_mode == "fixed":
            actual_seed = requested_seed
        elif seed_mode == "increment":
            actual_seed = NIMHTTPInference._last_used_seed + 1
        elif seed_mode == "decrement":
            actual_seed = NIMHTTPInference._last_used_seed - 1
        else:
            import random
            actual_seed = random.randint(0, 2**32 - 1)
        actual_seed = max(0, min(actual_seed, 2**32 - 1))
        NIMHTTPInference._last_used_seed = actual_seed

        # Samples (batch size)
        try:
            samples = max(1, min(4, int(self.get_parameter_value("samples") or 1)))
        except Exception:
            samples = 1

        url = base.rstrip("/") + "/" + route.lstrip("/")
        headers: Dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data: Dict[str, Any] | None = None
        if method == "POST":
            if isinstance(raw, str) and raw.strip().startswith("{"):
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {}
            else:
                data = {
                    "prompt": prompt,
                    "mode": (defaults.get("mode") if isinstance(defaults, dict) else "base"),
                    "seed": int(actual_seed),
                    "steps": int((defaults or {}).get("steps", 50)),
                    "samples": int(samples),
                }

        resp = requests.request(method, url, headers=headers, json=data if method == "POST" else None, timeout=300)
        try:
            js = resp.json()
        except Exception:
            js = {"status": resp.status_code, "text": resp.text}
        self.parameter_output_values["response"] = js
        # Output actual used seed on the same parameter
        try:
            self.parameter_output_values["seed"] = actual_seed
        except Exception:
            pass

        # Try to extract first base64 image and save via StaticFilesManager so UI renders
        try:
            arts = js.get("artifacts") or []
            saved_images = []
            if isinstance(arts, list) and arts:
                for idx, art in enumerate(arts[:4], start=1):
                    item = art if isinstance(art, dict) else {}
                    b64 = item.get("base64")
                    mime = (item.get("mime_type") or item.get("mime") or "").lower()
                    if not isinstance(b64, str) or len(b64) == 0:
                        continue
                    image_bytes = base64.b64decode(b64)

                    ext = ".jpg"
                    if "png" in mime:
                        ext = ".png"
                    elif "webp" in mime:
                        ext = ".webp"

                    route_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(self.get_parameter_value("route") or "/v1/infer")).strip("_")
                    filename = f"nim_{route_slug}_seed_{actual_seed}_{idx}_{int(time.time()*1000)}{ext}"

                    try:
                        static_url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)
                    except Exception:
                        fd, path = tempfile.mkstemp(suffix=ext)
                        with os.fdopen(fd, "wb") as f:
                            f.write(image_bytes)
                        static_url = f"file://{path}"

                    try:
                        art_obj = ImageUrlArtifact(value=static_url, name=f"Image {idx}")
                    except Exception:
                        art_obj = ImageUrlArtifact(static_url)
                    saved_images.append(art_obj)

                # First image quick output
                if saved_images:
                    self.parameter_output_values["image"] = saved_images[0]

                # Grid list
                try:
                    self.parameter_output_values["images"] = saved_images
                except Exception:
                    pass

                # Positional outputs
                layout = ["image_1_1", "image_1_2", "image_2_1", "image_2_2"]
                for i, name in enumerate(layout):
                    val = saved_images[i] if i < len(saved_images) else None
                    self.parameter_output_values[name] = val
        except Exception:
            pass

        return TextArtifact(f"HTTP {resp.status_code}")

