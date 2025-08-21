import os
import json
import shlex
import subprocess
import time
from typing import Any, Dict, List
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options
from griptape.artifacts import TextArtifact
import requests

class NIMContainerManager(ControlNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "NIM"
        self.description = "Start/stop/status for a NIM Docker/Podman container (GPU via WSL2/Docker Desktop)"

        # Inputs / properties
        self.add_parameter(Parameter(name="action", type="str", default_value="start", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["start","stop","status","restart","remove","inventory"])}, tooltip="Container lifecycle action."))
        self.add_parameter(Parameter(name="engine", type="str", default_value="docker", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["docker","podman"])}, tooltip="Container engine to use."))
        self.add_parameter(Parameter(name="name", type="str", default_value="nim", allowed_modes={ParameterMode.PROPERTY}, tooltip="Container name."))
        self.add_parameter(Parameter(name="image", type="str", default_value="", allowed_modes={ParameterMode.PROPERTY}, tooltip="Image (e.g., nvcr.io/nim/black-forest-labs/flux.1-dev)."))
        self.add_parameter(Parameter(name="tag", type="str", default_value="latest", allowed_modes={ParameterMode.PROPERTY}, tooltip="Image tag (e.g., 1.1.0)."))
        self.add_parameter(Parameter(name="ports", type="str", default_value="8000:8000", allowed_modes={ParameterMode.PROPERTY}, tooltip="Comma-separated host:container ports (e.g. 8000:8000)."))
        self.add_parameter(Parameter(name="shm_size", type="str", default_value="16g", allowed_modes={ParameterMode.PROPERTY}, tooltip="--shm-size value (e.g., 16g)."))
        self.add_parameter(Parameter(name="ipc_host", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, tooltip="Add --ipc=host to improve Triton readiness on WSL2."))
        self.add_parameter(Parameter(name="ngc_api_key", type="str", default_value=os.getenv("NGC_API_KEY", ""), allowed_modes={ParameterMode.PROPERTY}, ui_options={"display_name": "NGC API Key", "secret": True}, tooltip="NGC API key for private pulls or runtime use."))
        self.add_parameter(Parameter(name="pass_hf_env", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, tooltip="Pass host HF tokens (HUGGINGFACE_HUB_ACCESS_TOKEN/HF_TOKEN) into container."))
        self.add_parameter(Parameter(name="nim_model_variant", type="str", default_value="base", allowed_modes={ParameterMode.PROPERTY}, traits={Options(choices=["base","canny","depth","base+depth","base+canny","canny+depth"])}, tooltip="NIM model variant (NIM_MODEL_VARIANT)."))
        # New: explicit toggle for cache mount
        self.add_parameter(Parameter(name="use_cache_mount", type="bool", default_value=False, allowed_modes={ParameterMode.PROPERTY}, tooltip="Enable host cache bind-mount to /opt/nim/.cache/"))
        self.add_parameter(Parameter(name="cache_dir", type="str", default_value=os.path.expanduser("~/.cache/nim"), allowed_modes={ParameterMode.PROPERTY}, tooltip="Host cache directory to bind to /opt/nim/.cache/ (set 'use_cache_mount' to True)", ui_options={"hide_when": {"use_cache_mount": [False]}}))
        self.add_parameter(Parameter(name="env_extra", type="str", default_value="{}", allowed_modes={ParameterMode.PROPERTY}, tooltip="JSON object of additional environment variables."))
        self.add_parameter(Parameter(name="health_wait", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, tooltip="Wait for /v1/health to be OK before returning."))
        self.add_parameter(Parameter(name="health_timeout_s", type="int", default_value=300, allowed_modes={ParameterMode.PROPERTY}, tooltip="Max seconds to wait for health OK."))
        self.add_parameter(Parameter(name="base_url", type="str", default_value="http://localhost:8000", allowed_modes={ParameterMode.PROPERTY}, tooltip="Service base URL for health and outputs."))
        self.add_parameter(Parameter(name="detach", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, tooltip="Run container in background (-d)."))
        self.add_parameter(Parameter(name="add_ulimits", type="bool", default_value=True, allowed_modes={ParameterMode.PROPERTY}, tooltip="Add --ulimit memlock/stack for Triton stability."))
        self.add_parameter(Parameter(name="logs_tail", type="int", default_value=200, allowed_modes={ParameterMode.PROPERTY}, tooltip="Tail N lines from container logs after start/status."))

        # Outputs
        self.add_parameter(Parameter(name="container_id", output_type="str", allowed_modes={ParameterMode.OUTPUT}, ui_options={"display_name": "Container ID", "hide_property": True}, tooltip="Container ID or last run output."))
        self.add_parameter(Parameter(name="status", output_type="str", allowed_modes={ParameterMode.OUTPUT}, ui_options={"display_name": "Status", "hide_property": True}, tooltip="Human-readable status or error."))
        self.add_parameter(Parameter(name="logs", output_type="str", allowed_modes={ParameterMode.OUTPUT}, ui_options={"display_name": "Logs", "hide_property": True}, tooltip="Recent container logs (tail)."))
        self.add_parameter(Parameter(name="service_config", output_type="dict", allowed_modes={ParameterMode.OUTPUT}, ui_options={"display_name": "Service Config", "hide_property": True}, tooltip="Dict with base_url, route, defaults for inference."))

    def process(self) -> Any:
        yield lambda: self._run()

    def _run(self):
        action = (self.get_parameter_value("action") or "start").strip().lower()
        engine = (self.get_parameter_value("engine") or "docker").strip().lower()
        name = (self.get_parameter_value("name") or "nim").strip()
        image = (self.get_parameter_value("image") or "").strip()
        tag = (self.get_parameter_value("tag") or "latest").strip()
        ports = (self.get_parameter_value("ports") or "8000:8000").strip()
        shm_size = (self.get_parameter_value("shm_size") or "16g").strip()
        ngc_key = (self.get_parameter_value("ngc_api_key") or "").strip()
        pass_hf = bool(self.get_parameter_value("pass_hf_env"))
        nim_variant = (self.get_parameter_value("nim_model_variant") or "base").strip()
        use_cache_mount = bool(self.get_parameter_value("use_cache_mount"))
        cache_dir = (self.get_parameter_value("cache_dir") or "").strip()
        env_extra_raw = self.get_parameter_value("env_extra") or "{}"
        health_wait = bool(self.get_parameter_value("health_wait"))
        health_timeout_s = int(self.get_parameter_value("health_timeout_s") or 300)
        base_url = (self.get_parameter_value("base_url") or "http://localhost:8000").strip()
        detach = bool(self.get_parameter_value("detach"))
        ipc_host = bool(self.get_parameter_value("ipc_host"))
        logs_tail = int(self.get_parameter_value("logs_tail") or 200)
        add_ulimits = bool(self.get_parameter_value("add_ulimits"))

        def _run_cmd(args: List[str]) -> subprocess.CompletedProcess:
            return subprocess.run(args, capture_output=True, text=True)

        def _ce(*args: str) -> subprocess.CompletedProcess:
            return _run_cmd([engine, *args])

        # Ensure engine is available
        chk = _ce("version")
        if chk.returncode != 0:
            out = chk.stderr.strip() or chk.stdout.strip()
            self.parameter_output_values["status"] = f"{engine} unavailable: {out[:200]}"
            self.parameter_output_values["logs"] = out
            return TextArtifact(f"{engine} unavailable")

        if action == "start":
            if not image:
                self.parameter_output_values["status"] = "missing image (nvcr.io/...)"
                return TextArtifact("missing image")
            imgref = image if ":" in image else f"{image}:{tag}"
            # Build env
            env_pairs: Dict[str,str] = {}
            if ngc_key:
                env_pairs["NGC_API_KEY"] = ngc_key
            # Get HF token via standard service like other nodes
            token = None
            try:
                token = self.get_config_value(service="Huggingface", value="HUGGINGFACE_HUB_ACCESS_TOKEN")
            except Exception:
                token = None
            if not token:
                try:
                    token = self.get_config_value(service="Huggingface", value="HF_TOKEN")
                except Exception:
                    token = None
            if not token:
                token = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN") or os.getenv("HF_TOKEN")
            if pass_hf and token:
                env_pairs["HF_TOKEN"] = token
                env_pairs["HUGGINGFACE_HUB_ACCESS_TOKEN"] = token
            if nim_variant:
                env_pairs["NIM_MODEL_VARIANT"] = nim_variant
            try:
                extra = json.loads(env_extra_raw) if isinstance(env_extra_raw, str) else {}
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        if isinstance(k, str) and isinstance(v, (str,int,float,bool)):
                            env_pairs[k] = str(v)
            except Exception:
                pass
            # Optional cache mount (only if toggled)
            vol_args: List[str] = []
            if use_cache_mount:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                except Exception:
                    pass
                host_path = os.path.normpath(cache_dir) if cache_dir else os.path.expanduser("~/.cache/nim")
                vol_args = ["-v", f"{host_path}:/opt/nim/.cache/"]
            # Build port maps
            port_args: List[str] = []
            for p in str(ports).split(","):
                p = p.strip()
                if not p:
                    continue
                if ":" in p:
                    port_args += ["-p", p]
            run_args: List[str] = ["run"]
            # Engine-specific GPU flag
            if engine == "docker":
                run_args += ["--gpus", "all"]
            else:
                run_args += ["--device", "nvidia.com/gpu=all"]
            run_args += [
                "--name", name,
                "--shm-size", shm_size,
            ]
            if ipc_host:
                run_args += ["--ipc", "host"]
            if detach:
                run_args.append("-d")
            if add_ulimits:
                run_args += ["--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]
            for k, v in env_pairs.items():
                run_args += ["-e", f"{k}={v}"]
            run_args += port_args
            run_args += vol_args
            run_args += [imgref]
            res = _ce(*run_args)
            out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            # Mask token presence
            has_hf = bool(env_pairs.get("HF_TOKEN") or env_pairs.get("HUGGINGFACE_HUB_ACCESS_TOKEN"))
            out = (out or "") + f"\n[env] HF_TOKEN={'set' if has_hf else 'missing'}; NGC_API_KEY={'set' if bool(ngc_key) else 'missing'}; VARIANT={nim_variant}; CACHE={'on' if use_cache_mount else 'off'}"
            self.parameter_output_values["logs"] = out
            cid = (res.stdout or res.stderr).strip().splitlines()[-1] if (res.stdout or res.stderr) else ""
            self.parameter_output_values["container_id"] = cid
            if res.returncode != 0:
                self.parameter_output_values["status"] = f"start failed: {(res.stderr or res.stdout)[:500]}"
                return TextArtifact("start failed")
            # Confirm container present and port exposed
            ps = _ce("ps", "-a", "--filter", f"name={name}", "--format", "{{.ID}} {{.Image}} {{.Status}} {{.Ports}}")
            self.parameter_output_values["logs"] += "\n" + (ps.stdout or ps.stderr)
            # Health wait (HTTP) with log warmup fallback
            status_txt = "started"
            if health_wait:
                t0 = time.time()
                ok = False
                while time.time() - t0 < health_timeout_s:
                    try:
                        r = requests.get(base_url.rstrip("/") + "/v1/health/ready", timeout=5)
                        if r.status_code == 200:
                            ok = True
                            break
                    except Exception:
                        pass
                    time.sleep(3)
                status_txt = "running" if ok else "started (health timeout)"
            self.parameter_output_values["status"] = status_txt
            # Emit service_config for inference node
            self.parameter_output_values["service_config"] = {
                "base_url": base_url,
                "route": "/v1/infer",
                "defaults": {"mode": nim_variant or "base", "steps": 50, "seed": 0}
            }
            return TextArtifact(status_txt)

        if action == "stop":
            res = _ce("stop", name)
            self.parameter_output_values["status"] = "stopped" if res.returncode == 0 else f"stop failed: {(res.stderr or res.stdout)[:200]}"
            return TextArtifact("stopped" if res.returncode == 0 else "stop failed")

        if action == "remove":
            res = _ce("rm", "-f", name)
            self.parameter_output_values["status"] = "removed" if res.returncode == 0 else f"remove failed: {(res.stderr or res.stdout)[:200]}"
            return TextArtifact("removed" if res.returncode == 0 else "remove failed")

        if action == "restart":
            _ce("stop", name)
            res = _ce("start", name)
            self.parameter_output_values["status"] = "running" if res.returncode == 0 else f"restart failed: {(res.stderr or res.stdout)[:200]}"
            lg = _ce("logs", "--tail", str(logs_tail), name)
            self.parameter_output_values["logs"] = (lg.stdout or lg.stderr)
            return TextArtifact("restarted" if res.returncode == 0 else "restart failed")

        if action == "inventory":
            # List images and containers; try /v1/metadata for running nim
            imgs = _ce("images", "nvcr.io", "--format", "{{.Repository}}\t{{.Tag}}\t{{.Size}}")
            ps = _ce("ps", "--format", "{{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}")
            report = ["[Images]", imgs.stdout or imgs.stderr, "\n[Containers]", ps.stdout or ps.stderr]
            try:
                r = requests.get(base_url.rstrip("/") + "/v1/metadata", timeout=3)
                if r.ok:
                    report.append("\n[Metadata]")
                    report.append(r.text)
            except Exception:
                pass
            out = "\n".join([s for s in report if s is not None])
            self.parameter_output_values["logs"] = out
            self.parameter_output_values["status"] = "inventory"
            return TextArtifact("inventory")

        # status
        ps = _ce("ps", "-a", "--filter", f"name={name}", "--format", "{{.ID}} {{.Image}} {{.Status}} {{.Ports}}")
        self.parameter_output_values["status"] = (ps.stdout or ps.stderr)
        lg = _ce("logs", "--tail", str(logs_tail), name)
        self.parameter_output_values["logs"] = (lg.stdout or lg.stderr)
        # Keep emitting current service_config for wiring flows
        self.parameter_output_values["service_config"] = {
            "base_url": base_url,
            "route": "/v1/infer",
        }
        return TextArtifact("status")
