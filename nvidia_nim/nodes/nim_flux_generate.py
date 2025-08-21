from typing import Any, Dict, List
import os
import json
import base64
import time
import re
import tempfile
import requests
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options
from griptape.artifacts import TextArtifact, ImageUrlArtifact
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class NIMFluxGenerate(ControlNode):
    _last_used_seed: int = 12345

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "NIM"
        self.description = "Flux text-to-image across dev/schnell/onyx with batching & seed control"

        default_base = os.getenv("NIM_BASE_URL", "http://localhost:8000")

        # Inputs/properties
        self.add_parameter(Parameter(name="service_config", input_types=["dict"], type="dict", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}, ui_options={"display_name": "Service Config", "hide_property": True}, tooltip="Config from container (base_url, route, defaults)."))
        self.add_parameter(Parameter(name="base_url", type="str", default_value=default_base, allowed_modes={ParameterMode.PROPERTY}, tooltip="Base URL of the NIM service."))
        self.add_parameter(Parameter(name="route", type="str", default_value="/v1/infer", allowed_modes={ParameterMode.PROPERTY}, tooltip="Inference route (e.g., /v1/infer)."))

        self.add_parameter(Parameter(name="mode", type="str", default_value="base", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["base", "dev", "schnell", "onyx"])}, tooltip="Flux variant/mode."))
        self.add_parameter(Parameter(name="prompt", input_types=["str"], type="str", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}, tooltip="Text prompt."))
        self.add_parameter(Parameter(name="width", type="int", default_value=1024, allowed_modes={ParameterMode.PROPERTY}, tooltip="Image width."))
        self.add_parameter(Parameter(name="height", type="int", default_value=1024, allowed_modes={ParameterMode.PROPERTY}, tooltip="Image height."))
        self.add_parameter(Parameter(name="steps", type="int", default_value=50, allowed_modes={ParameterMode.PROPERTY}, tooltip="Inference steps."))
        self.add_parameter(Parameter(name="cfg_scale", type="float", default_value=3.5, allowed_modes={ParameterMode.PROPERTY}, tooltip="CFG guidance scale."))
        self.add_parameter(Parameter(name="samples", type="int", default_value=1, allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=[1, 2, 3, 4])}, tooltip="Number of images to generate (1-4)."))

        # Seed controls (in/out)
        self.add_parameter(Parameter(name="seed", type="int", default_value=12345, allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT, ParameterMode.OUTPUT}, tooltip="Seed for reproducibility (outputs actual used seed)."))
        self.add_parameter(Parameter(name="seed_control", type="str", default_value="randomize", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["fixed", "increment", "decrement", "randomize"])}, tooltip="Seed control mode."))

        # Outputs (grouped)
        from griptape_nodes.exe_types.core_types import ParameterGroup
        with ParameterGroup(name="Images") as images_group:
            self.add_parameter(Parameter(
                name="images",
                type="list",
                default_value=[],
                output_type="list[ImageUrlArtifact]",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display": "grid", "columns": 2},
                tooltip="Generated images (grid)."
            ))
            self.add_parameter(Parameter(name="image_1_1", type="ImageUrlArtifact", output_type="ImageUrlArtifact", allowed_modes={ParameterMode.OUTPUT}, ui_options={"hide_property": True}, tooltip="Image [1,1]."))
            self.add_parameter(Parameter(name="image_1_2", type="ImageUrlArtifact", output_type="ImageUrlArtifact", allowed_modes={ParameterMode.OUTPUT}, ui_options={"hide_property": True, "hide_when": {"samples": [1]}}, tooltip="Image [1,2]."))
            self.add_parameter(Parameter(name="image_2_1", type="ImageUrlArtifact", output_type="ImageUrlArtifact", allowed_modes={ParameterMode.OUTPUT}, ui_options={"hide_property": True, "hide_when": {"samples": [1, 2]}}, tooltip="Image [2,1]."))
            self.add_parameter(Parameter(name="image_2_2", type="ImageUrlArtifact", output_type="ImageUrlArtifact", allowed_modes={ParameterMode.OUTPUT}, ui_options={"hide_property": True, "hide_when": {"samples": [1, 2, 3]}}, tooltip="Image [2,2]."))
        self.add_node_element(images_group)
        self.add_parameter(Parameter(name="response", output_type="dict", allowed_modes={ParameterMode.OUTPUT}, ui_options={"display_name": "Response", "hide_property": True}, tooltip="Raw JSON response."))
        self.add_parameter(Parameter(name="performance", output_type="json", allowed_modes={ParameterMode.OUTPUT}, ui_options={"hide_property": True}, tooltip="Timing and settings per run."))

        # Ensure initial visibility matches default samples
        try:
            default_samples = int(self.get_parameter_value("samples") or 1)
            self._apply_positional_visibility(default_samples)
        except Exception:
            pass

    def process(self) -> Any:
        yield lambda: self._run()

    def _resolve(self, key: str, svc: Dict[str, Any], fallback: Any) -> Any:
        # Prefer service_config over node defaults to honor orchestrator wiring
        try:
            if isinstance(svc, dict) and svc.get(key) not in (None, ""):
                return svc.get(key)
            val = self.get_parameter_value(key)
            return val if val not in (None, "") else fallback
        except Exception:
            return fallback

    def _run(self):
        svc = self.get_parameter_value("service_config") or {}
        base = self._resolve("base_url", svc, "http://localhost:8000")
        route = self._resolve("route", svc, "/v1/infer")
        prompt = self.get_parameter_value("prompt") or "A simple coffee shop interior"
        mode = self.get_parameter_value("mode") or (svc.get("defaults", {}).get("mode") if isinstance(svc, dict) else "base") or "base"
        width = int(self.get_parameter_value("width") or 1024)
        height = int(self.get_parameter_value("height") or 1024)
        steps = int(self.get_parameter_value("steps") or int((svc.get("defaults", {}) if isinstance(svc, dict) else {}).get("steps", 50)))
        cfg_scale = float(self.get_parameter_value("cfg_scale") or 3.5)
        samples = max(1, min(4, int(self.get_parameter_value("samples") or 1)))

        # Seed control
        try:
            requested_seed = int(self.get_parameter_value("seed") or 12345)
        except Exception:
            requested_seed = 12345
        seed_mode = (self.get_parameter_value("seed_control") or "randomize").lower()
        if seed_mode == "fixed":
            actual_seed = requested_seed
        elif seed_mode == "increment":
            actual_seed = NIMFluxGenerate._last_used_seed + 1
        elif seed_mode == "decrement":
            actual_seed = NIMFluxGenerate._last_used_seed - 1
        else:
            import random
            actual_seed = random.randint(0, 2**32 - 1)
        actual_seed = max(0, min(actual_seed, 2**32 - 1))
        NIMFluxGenerate._last_used_seed = actual_seed
        self.parameter_output_values["seed"] = actual_seed

        base_norm = str(base or "").replace("localhost", "127.0.0.1")
        url = base_norm.rstrip("/") + "/" + route.lstrip("/")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        def _do_request(seed_val: int) -> Dict[str, Any]:
            payload = {
                "prompt": prompt,
                "mode": mode,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                # Many NIMs only allow samples<=1; emulate batching client-side
                "samples": 1,
                "seed": int(seed_val),
            }
            r = requests.post(url, headers=headers, json=payload, timeout=600)
            try:
                return r.json()
            except Exception:
                return {"status": r.status_code, "text": r.text}

        all_images: List[ImageUrlArtifact] = []
        seeds_used: List[int] = []
        per_call: List[Dict[str, Any]] = []
        t_start = time.perf_counter()
        # initial (no status output parameter per spec)
        # Always loop client-side if samples>1
        loop_count = samples if samples > 1 else 1
        current_seed = actual_seed
        for i in range(loop_count):
            t0 = time.perf_counter()
            js = _do_request(current_seed)
            dur = time.perf_counter() - t0
            # Save last response for debugging
            self.parameter_output_values["response"] = js
            per_call.append({
                "index": i + 1,
                "seed": int(current_seed),
                "duration_s": round(dur, 3),
                "status": js.get("status", 200) if isinstance(js, dict) else 200,
            })
            # no status publishing
            # Extract artifacts
            try:
                arts = js.get("artifacts") or []
                for idx, art in enumerate(arts[:1], start=1):  # keep one per call for grid consistency
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
                    filename = f"nim_flux_{route_slug}_seed_{current_seed}_{int(time.time()*1000)}{ext}"
                    try:
                        static_url = GriptapeNodes.StaticFilesManager().save_static_file(data, filename)
                    except Exception:
                        fd, path = tempfile.mkstemp(suffix=ext)
                        with os.fdopen(fd, "wb") as f:
                            f.write(data)
                        static_url = f"file://{path}"
                    try:
                        img = ImageUrlArtifact(value=static_url, name=f"Image {len(all_images)+1}")
                    except Exception:
                        img = ImageUrlArtifact(static_url)
                    all_images.append(img)
                    seeds_used.append(current_seed)
            except Exception:
                pass
            # Advance seed based on control for next iteration
            if loop_count > 1:
                if seed_mode == "fixed":
                    pass
                elif seed_mode == "increment":
                    current_seed = (current_seed + 1) % (2**32)
                elif seed_mode == "decrement":
                    current_seed = (current_seed - 1) % (2**32)
                else:
                    import random
                    current_seed = random.randint(0, 2**32 - 1)

        # Outputs
        images = all_images[:4]
        try:
            self.parameter_output_values["images"] = images
        except Exception:
            pass
        # Positional outputs (for wiring into downstream nodes like Kontext)
        layout = ["image_1_1", "image_1_2", "image_2_1", "image_2_2"]
        for i, name in enumerate(layout):
            val = images[i] if i < len(images) else None
            self.parameter_output_values[name] = val

        # Performance summary
        total_s = time.perf_counter() - t_start
        perf = {
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "mode": mode,
            "samples_requested": samples,
            "seeds": seeds_used,
            "per_call": per_call,
            "total_s": round(total_s, 3),
        }
        try:
            self.parameter_output_values["performance"] = perf
        except Exception:
            pass

        # Derive an HTTP-like status from last call for UI summary
        try:
            last_status = per_call[-1]["status"] if per_call else (self.parameter_output_values.get("response", {}) or {}).get("status", 200)  # type: ignore
        except Exception:
            last_status = 200
        return TextArtifact(f"HTTP {last_status}")

    # Visibility controls like other working nodes
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


