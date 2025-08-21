import os
import json
import shlex
import subprocess
import time
from typing import Any, Dict, List
import re
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options
from griptape.artifacts import TextArtifact


class NIMOrchestrator(ControlNode):
    """Minimal orchestrator for NIM with auto-inventory.

    - Auto-discovers images and containers and populates dropdowns.
    - start: launches selected image on chosen port and waits for health (optional).
    - stop: stops selected container.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "NIM"
        self.description = "Start/stop NIMs with auto-inventory dropdowns"

        # Inputs (minimal)
        self.add_parameter(Parameter(name="action", type="str", default_value="start", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["start","stop","status"])}, tooltip="Lifecycle action (start/stop/status)."))
        self.add_parameter(Parameter(name="engine", type="str", default_value="docker", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["docker","podman"])}, ui_options={"hide_property": True}, tooltip="Container engine (hidden)."))

        # Initial inventory for dropdowns
        containers, images = self._inventory_once("docker")
        image_choices: List[str] = images or [
            "nvcr.io/nim/black-forest-labs/flux.1-dev:latest",
            "nvcr.io/nim/black-forest-labs/flux.1-kontext-dev:latest",
            "nvcr.io/nim/stabilityai/stable-diffusion-3.5-large:latest",
        ]
        container_choices: List[str] = containers or ["nim", "kontext"]
        self.add_parameter(Parameter(name="target_image", type="str", default_value=image_choices[0], allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=image_choices)}, tooltip="NIM image to start (repo:tag)."))
        self.add_parameter(Parameter(name="target_container", type="str", default_value=container_choices[0], allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=container_choices)}, tooltip="Container to stop."))

        # Start parameters
        self.add_parameter(Parameter(name="name", type="str", default_value="nim", allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="Container name (hidden; auto)."))
        self.add_parameter(Parameter(name="host_port", type="int", default_value=8000, allowed_modes={ParameterMode.PROPERTY}, tooltip="Host port to publish (maps to 8000 in-container)."))
        self.add_parameter(Parameter(name="shm_size", type="str", default_value="16g", allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="--shm-size value (hidden)."))
        self.add_parameter(Parameter(name="ipc_host", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="Add --ipc=host (hidden)."))
        self.add_parameter(Parameter(name="add_ulimits", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="Add --ulimit memlock/stack (hidden)."))
        self.add_parameter(Parameter(name="pass_hf_env", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="Pass HF token (hidden)."))
        self.add_parameter(Parameter(name="ngc_api_key", type="str", default_value=os.getenv("NGC_API_KEY", ""), allowed_modes={ParameterMode.PROPERTY}, ui_options={"display_name": "NGC API Key", "secret": True, "hide_property": True}, tooltip="NGC API key (hidden)."))
        self.add_parameter(Parameter(name="nim_model_variant", type="str", default_value="base", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["base","canny","depth","base+depth","base+canny","canny+depth"])}, ui_options={"hide_property": True}, tooltip="NIM_MODEL_VARIANT (hidden)."))
        self.add_parameter(Parameter(name="use_cache_mount", type="bool", default_value=False, allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="Bind mount cache (hidden)."))
        self.add_parameter(Parameter(name="cache_dir", type="str", default_value=os.path.expanduser("~/.cache/nim"), allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="Cache dir (hidden)."))
        self.add_parameter(Parameter(name="health_wait", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="Health wait (hidden)."))
        self.add_parameter(Parameter(name="health_timeout_s", type="int", default_value=600, allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="Health timeout (hidden)."))
        self.add_parameter(Parameter(name="detach", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, ui_options={"hide_property": True}, tooltip="Detach (-d) (hidden)."))

        # Outputs
        self.add_parameter(Parameter(name="status", output_type="str", allowed_modes={ParameterMode.OUTPUT}, ui_options={"hide_property": True}, tooltip="Status."))
        self.add_parameter(Parameter(name="logs", output_type="str", allowed_modes={ParameterMode.OUTPUT}, ui_options={"hide_property": True}, tooltip="Logs."))
        self.add_parameter(Parameter(name="service_config", output_type="dict", allowed_modes={ParameterMode.OUTPUT}, ui_options={"hide_property": True}, tooltip="Service config for inference nodes."))

    def process(self) -> Any:
        yield lambda: self._run()

    # Helpers
    def _run_cmd(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True)

    def _ce(self, engine: str, *args: str) -> subprocess.CompletedProcess:
        return self._run_cmd([engine, *args])

    def _inventory_once(self, engine: str) -> (List[str], List[str]):
        containers: List[str] = []
        images: List[str] = []
        try:
            # Include all containers (running or stopped)
            ps = self._ce(engine, "ps", "-a", "--format", "{{.Names}}")
            if ps.returncode == 0:
                seen = set()
                for line in (ps.stdout or "").splitlines():
                    name = line.strip()
                    if name and name not in seen:
                        containers.append(name)
                        seen.add(name)
        except Exception:
            pass
        try:
            # List all images, then filter to nvcr.io registries (works across repos like stabilityai, bfl, etc.)
            im = self._ce(engine, "images", "--format", "{{.Repository}}:{{.Tag}}")
            if im.returncode == 0:
                for line in (im.stdout or "").splitlines():
                    ref = line.strip()
                    if not ref:
                        continue
                    # Only keep NGC images by default; drop dangling <none> tags
                    if ref.startswith("nvcr.io/") and not ref.endswith(":<none>"):
                        images.append(ref)
        except Exception:
            pass
        return containers, images

    def _container_exists(self, engine: str, name: str) -> bool:
        try:
            ps = self._ce(engine, "ps", "-a", "--format", "{{.Names}}")
            if ps.returncode == 0:
                for line in (ps.stdout or "").splitlines():
                    if line.strip() == name:
                        return True
        except Exception:
            pass
        return False

    def _container_running(self, engine: str, name: str) -> bool:
        try:
            ps = self._ce(engine, "ps", "--format", "{{.Names}}")
            if ps.returncode == 0:
                for line in (ps.stdout or "").splitlines():
                    if line.strip() == name:
                        return True
        except Exception:
            pass
        return False

    def _container_host_port(self, engine: str, name: str, fallback: int) -> int:
        try:
            ps = self._ce(engine, "ps", "--filter", f"name={name}", "--format", "{{.Ports}}")
            if ps.returncode == 0:
                s = (ps.stdout or "").strip()
                m = re.search(r":(\d+)->8000/", s)
                if m:
                    return int(m.group(1))
        except Exception:
            pass
        try:
            fmt = "{{ (index (index .NetworkSettings.Ports \"8000/tcp\") 0).HostPort }}"
            ins = self._ce(engine, "inspect", "-f", fmt, name)
            if ins.returncode == 0:
                hp = (ins.stdout or "").strip()
                if hp.isdigit():
                    return int(hp)
        except Exception:
            pass
        return int(fallback)

    def _hf_token(self) -> str | None:
        try:
            t = self.get_config_value(service="Huggingface", value="HUGGINGFACE_HUB_ACCESS_TOKEN")
        except Exception:
            t = None
        if not t:
            try:
                t = self.get_config_value(service="Huggingface", value="HF_TOKEN")
            except Exception:
                t = None
        return t or os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

    def _run(self):
        action = (self.get_parameter_value("action") or "start").lower()
        engine = (self.get_parameter_value("engine") or "docker").lower()
        name = (self.get_parameter_value("name") or "nim").strip()
        # If user selected a target_container, prefer it as the container name for start
        selected_target_name = (self.get_parameter_value("target_container") or "").strip()
        if (self.get_parameter_value("action") or "start").lower() == "start" and selected_target_name:
            name = selected_target_name
        host_port = int(self.get_parameter_value("host_port") or 8000)
        shm_size = (self.get_parameter_value("shm_size") or "16g").strip()
        pass_hf = bool(self.get_parameter_value("pass_hf_env"))
        ngc_key = (self.get_parameter_value("ngc_api_key") or "").strip()
        variant = (self.get_parameter_value("nim_model_variant") or "base").strip()
        use_cache = bool(self.get_parameter_value("use_cache_mount"))
        cache_dir = (self.get_parameter_value("cache_dir") or os.path.expanduser("~/.cache/nim")).strip()
        health_wait = bool(self.get_parameter_value("health_wait"))
        health_timeout_s = int(self.get_parameter_value("health_timeout_s") or 600)
        detach = bool(self.get_parameter_value("detach"))
        ipc_host = bool(self.get_parameter_value("ipc_host"))
        add_ulimits = bool(self.get_parameter_value("add_ulimits"))

        # Auto-inventory and update dropdowns
        containers, images = self._inventory_once(engine)
        try:
            if images:
                self.update_parameter_traits("target_image", {Options(choices=images)})
            if containers:
                self.update_parameter_traits("target_container", {Options(choices=containers)})
        except Exception:
            pass

        if action == "start":
            imgref = (self.get_parameter_value("target_image") or "").strip()
            env_pairs: Dict[str, str] = {}
            if ngc_key:
                env_pairs["NGC_API_KEY"] = ngc_key
            if pass_hf:
                tok = self._hf_token()
                if tok:
                    env_pairs["HF_TOKEN"] = tok
                    env_pairs["HUGGINGFACE_HUB_ACCESS_TOKEN"] = tok
            if variant:
                env_pairs["NIM_MODEL_VARIANT"] = variant

            if self._container_exists(engine, name):
                res = self._ce(engine, "start", name)
                out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
                self.parameter_output_values["logs"] = out
                if res.returncode != 0 and not self._container_running(engine, name):
                    self.parameter_output_values["status"] = "start failed"
                    return TextArtifact("start failed")
            else:
                run_args: List[str] = ["run"]
                if engine == "docker":
                    run_args += ["--gpus", "all"]
                else:
                    run_args += ["--device", "nvidia.com/gpu=all"]
                run_args += ["--name", name, "--shm-size", shm_size]
                if ipc_host:
                    run_args += ["--ipc", "host"]
                if detach:
                    run_args.append("-d")
                if add_ulimits:
                    run_args += ["--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]
                run_args += ["-p", f"{host_port}:8000"]
                if use_cache:
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                    except Exception:
                        pass
                run_args += ["-v", f"{os.path.normpath(cache_dir)}:/opt/nim/.cache/"] if use_cache else []
                for k, v in env_pairs.items():
                    run_args += ["-e", f"{k}={v}"]
                run_args += [imgref]
                res = self._ce(engine, *run_args)
                out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
                self.parameter_output_values["logs"] = out
                if res.returncode != 0:
                    self.parameter_output_values["status"] = "start failed"
                    return TextArtifact("start failed")
            # Health wait
            mapped = self._container_host_port(engine, name, host_port)
            base_url = f"http://127.0.0.1:{mapped}"
            if health_wait:
                ok = False
                t0 = time.time()
                while time.time() - t0 < health_timeout_s:
                    try:
                        import requests
                        r = requests.get(base_url + "/v1/health/ready", timeout=5)
                        if r.status_code == 200:
                            ok = True
                            break
                    except Exception:
                        pass
                    time.sleep(3)
                self.parameter_output_values["status"] = "running" if ok else "started (health timeout)"
            else:
                self.parameter_output_values["status"] = "started"
            self.parameter_output_values["service_config"] = {"base_url": base_url, "route": "/v1/infer", "defaults": {"mode": variant or "base"}}
            return TextArtifact("started")

        if action == "stop":
            target_name = (self.get_parameter_value("target_container") or name).strip()
            res = self._ce(engine, "stop", target_name)
            self.parameter_output_values["status"] = "stopped" if res.returncode == 0 else "stop failed"
            self.parameter_output_values["logs"] = (res.stdout or res.stderr)
            return TextArtifact("stopped" if res.returncode == 0 else "stop failed")

        if action == "status":
            mapped = self._container_host_port(engine, name, host_port)
            base_url = f"http://127.0.0.1:{mapped}"
            running = self._container_running(engine, name)
            self.parameter_output_values["status"] = "running" if running else ("stopped" if self._container_exists(engine, name) else "absent")
            lg = self._ce(engine, "logs", "--tail", "50", name)
            self.parameter_output_values["logs"] = (lg.stdout or lg.stderr)
            self.parameter_output_values["service_config"] = {"base_url": base_url, "route": "/v1/infer"}
            return TextArtifact(self.parameter_output_values["status"])

        self.parameter_output_values["status"] = f"unknown action: {action}"
        return TextArtifact("error")


