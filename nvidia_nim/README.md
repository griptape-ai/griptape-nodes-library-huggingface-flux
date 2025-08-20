# NVIDIA NIM on Windows (WSL2 + Docker) - Quick Start

This guide shows how to run NVIDIA NIM locally on Windows using Docker Desktop with WSL2, then call it from a separate Griptape Nodes library (isolated from CUDA/Diffusers).

## 1) Prerequisites
- Windows 11
- NVIDIA RTX GPU with recent driver (R555+). Verify in PowerShell: `nvidia-smi`
- Admin access to install WSL and Docker Desktop
- NGC account + API key: `https://ngc.nvidia.com`

## 2) Enable WSL2 and install Ubuntu
Run PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu
```

You will see: "Changes will not be effective until the system is rebooted." Reboot Windows.

After reboot, install a specific distro name (avoid the generic alias):

```powershell
wsl --list --online
wsl --install -d Ubuntu-22.04
```

If you hit HCS_E_SERVICE_NOT_AVAILABLE or similar:

```powershell
# Ensure features are enabled (Admin PowerShell)
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All

# Ensure hypervisor and services
bcdedit /set hypervisorlaunchtype auto
net start vmcompute
# LxssManager is on-demand; it's OK if it doesn't start manually on newer builds

# BIOS/UEFI: ensure virtualization (Intel VT-x/AMD-V) is enabled
# Windows Security -> Device Security -> Core isolation: turn OFF Memory integrity if blocked

# Update and set v2
wsl --shutdown
wsl --update
wsl --set-default-version 2
```

To run commands in the installed distro (note the exact name):

```powershell
wsl -d Ubuntu-22.04 -- echo ok
```

## 3) Install Docker Desktop with WSL2 backend
- Install: `https://www.docker.com/products/docker-desktop/`
- Settings → General: enable "Use the WSL 2 based engine"
- Settings → Resources → WSL Integration: enable Ubuntu-22.04
- Apply & restart Docker Desktop

## 4) Verify GPU in WSL and Docker
WSL (the final relay error is benign):

```powershell
wsl -d Ubuntu-22.04 -- nvidia-smi
```

Docker (this is the real gate):

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Both should show your RTX 5090 and CUDA 12.8.

## 5) Login to NGC and run a NIM container
```powershell
docker login nvcr.io -u '$oauthtoken' -p <YOUR_NGC_API_KEY>

# Replace with the exact NIM image:tag from NGC Catalog (example shown)
# Optional: add --ipc=host on WSL2 to unblock Triton readiness
# Optional: mount cache to persist engine plans (Windows path must use forward slashes)
docker run --rm --gpus all --name nim -p 8000:8000 --shm-size=16g `
  --ipc=host `
  -e NGC_API_KEY=<YOUR_NGC_API_KEY> `
  -e HF_TOKEN=$Env:HUGGINGFACE_HUB_ACCESS_TOKEN `
  -e HUGGINGFACE_HUB_ACCESS_TOKEN=$Env:HUGGINGFACE_HUB_ACCESS_TOKEN `
  -e NIM_MODEL_VARIANT=base `
  -v C:/Users/<you>/.cache/nim:/opt/nim/.cache/ `
  nvcr.io/<org>/<nim-image>:<tag>
```

Health check and warmup:

```powershell
# Discover available routes
curl http://127.0.0.1:8000/openapi.json | cat

# Basic health
curl http://127.0.0.1:8000/v1/health | cat

# Readiness (will return ready only after Triton is up)
curl http://127.0.0.1:8000/v1/health/ready | cat

# Logs (look for: "Pipeline warmup: done")
docker logs --tail 200 nim | cat
```

## 6) Separate Griptape Nodes library (NIM)
This repo includes a separate library under `nvidia_nim/` that calls a NIM HTTP endpoint. Minimal deps, isolated from CUDA.

- Manifest: `nvidia_nim/griptape-nodes-library.json`
- Loader: `nvidia_nim/library_loader.py`
- Node: `nvidia_nim/nodes/nim_http.py`

Register the library in Griptape (Add by JSON path) and select `nvidia_nim/griptape-nodes-library.json`.

## 7) Use with the provided nodes

1) Start container (node): NIM → NIM Container Manager
- action: start
- name: nim
- image: nvcr.io/<org>/<nim-image> (or full image:tag)
- tag: latest (or specific)
- ports: 8000:8000
- shm_size: 16g
- ngc_api_key: <your NGC API key>
- pass_hf_env: true (for gated HF models)
- env_extra: {"NIM_MODEL_VARIANT":"base"}
- use_cache_mount: true
- cache_dir: C:\\Users\\<you>\\.cache\\nim
- ipc_host: true
- health_wait: true
- health_timeout_s: 600

Outputs:
- status/logs for visibility
- service_config (base_url, route=/v1/infer, defaults) for chaining

2) Inference call (node): NIM → NIM HTTP Inference
- service_config: connect from previous node
- base_url: http://127.0.0.1:8000 (optional override)
- route: /v1/infer
- method: POST
- json_payload:
```json
{"prompt":"A simple coffee shop interior","mode":"base","width":1024,"height":1024,"steps":50,"seed":0,"samples":1,"cfg_scale":3.5}
```

The node decodes artifacts[0].base64 and saves via Griptape StaticFilesManager; the `image` output is a UI-renderable URL. Fallback saves to a temp file.

## 8) Troubleshooting (expanded)
- GPU not visible in Docker: update NVIDIA driver (R555+), enable WSL2 backend + distro integration in Docker Desktop, verify `wsl -d Ubuntu-22.04 -- nvidia-smi` and `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`.
- docker: invalid reference format: check `image[:tag]` spelling and quoting; ensure Windows `-v` paths use forward slashes (e.g., `C:/Users/<you>/.cache/nim`).
- Container name conflict: `docker rm -f nim` or set a different `name`.
- Readiness UNAVAILABLE (Triton 8001): wait for warmup, use `--ipc=host`, optionally disable cache mount once (`use_cache_mount: false`) to force a clean engine build, watch logs for errors, re-check `/openapi.json` for real routes.
- 401/403 while pulling/first infer: pass `NGC_API_KEY`; for HF downloads, ensure `HF_TOKEN`/`HUGGINGFACE_HUB_ACCESS_TOKEN` is set/passed and you accepted the model license on Hugging Face.
- 404/Not Found: use correct routes (`/v1/health`, `/v1/health/ready`, `/v1/infer`). Discover via `/openapi.json`.
- Connection refused/aborted: service not ready or crashed; check `docker ps`, logs, increase HTTP timeout, prefer `http://127.0.0.1:8000`.
- 500 on POST: check container logs during the request; validate JSON schema (fields like `prompt`, `mode`, `width`, `height`, `steps`, `seed`, `samples`, `cfg_scale`).

## 9) Example: full docker run (mirrors node settings)
```powershell
docker run --rm --gpus all --name nim -p 8000:8000 `
  --shm-size=16g --ipc=host `
  -e NGC_API_KEY=<YOUR_NGC_API_KEY> `
  -e HF_TOKEN=$Env:HUGGINGFACE_HUB_ACCESS_TOKEN `
  -e HUGGINGFACE_HUB_ACCESS_TOKEN=$Env:HUGGINGFACE_HUB_ACCESS_TOKEN `
  -e NIM_MODEL_VARIANT=base `
  -v C:/Users/<you>/.cache/nim:/opt/nim/.cache/ `
  nvcr.io/<org>/<nim-image>:<tag>
```

## 10) Notes for Windows paths and ports
- Use forward slashes in `-v` bind mounts (`C:/Users/...`), not backslashes.
- If corporate proxy/DNS interferes, prefer `127.0.0.1` over `localhost`.

