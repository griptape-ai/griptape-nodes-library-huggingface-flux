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

        self.add_parameter(Parameter(name="response", output_type="dict", allowed_modes={ParameterMode.OUTPUT}, ui_options={"display_name": "Response", "hide_property": True}, tooltip="Raw JSON response."))
        self.add_parameter(Parameter(name="image", output_type="ImageUrlArtifact", allowed_modes={ParameterMode.OUTPUT}, tooltip="First generated image as file URL."))

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
                    "seed": int((defaults or {}).get("seed", 0)),
                    "steps": int((defaults or {}).get("steps", 50)),
                }

        resp = requests.request(method, url, headers=headers, json=data if method == "POST" else None, timeout=300)
        try:
            js = resp.json()
        except Exception:
            js = {"status": resp.status_code, "text": resp.text}
        self.parameter_output_values["response"] = js

        # Try to extract first base64 image and save via StaticFilesManager so UI renders
        try:
            arts = js.get("artifacts") or []
            if isinstance(arts, list) and arts:
                first = arts[0] if isinstance(arts[0], dict) else {}
                b64 = first.get("base64")
                mime = (first.get("mime_type") or first.get("mime") or "").lower()
                if isinstance(b64, str) and len(b64) > 0:
                    image_bytes = base64.b64decode(b64)

                    # Pick extension from mime, default to .jpg
                    ext = ".jpg"
                    if "png" in mime:
                        ext = ".png"
                    elif "webp" in mime:
                        ext = ".webp"

                    # Build a safe filename from route and time
                    route_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(self.get_parameter_value("route") or "/v1/infer")).strip("_")
                    filename = f"nim_{route_slug}_{int(time.time()*1000)}{ext}"

                    static_url = None
                    try:
                        static_url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)
                    except Exception:
                        # Fallback to temp file if static save fails
                        fd, path = tempfile.mkstemp(suffix=ext)
                        with os.fdopen(fd, "wb") as f:
                            f.write(image_bytes)
                        static_url = f"file://{path}"

                    try:
                        self.parameter_output_values["image"] = ImageUrlArtifact(value=static_url)
                    except Exception:
                        self.parameter_output_values["image"] = ImageUrlArtifact(static_url)
        except Exception:
            # Ignore image extraction errors; response still returned
            pass

        return TextArtifact(f"HTTP {resp.status_code}")

