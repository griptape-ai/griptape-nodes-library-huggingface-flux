import os
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelDiscovery:
    display_name: str
    path: Path
    repo_id: str


class HFCacheScanner:
    """Minimal HF cache scanner that does not hit the network.
    Walks HF_HUB_CACHE/models--*/snapshots/* to produce local snapshot paths.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        # Prefer official env var used by huggingface_hub, but also honor legacy vars
        primary = (
            cache_dir
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or os.environ.get("HF_HUB_CACHE")
            or os.path.expanduser("~/.cache/huggingface")
        )
        self.cache_dir = primary
        self._logger = logging.getLogger(__name__)

    def _iter_model_snapshots(self) -> List[tuple[str, Path]]:
        # Try common HF cache locations (Linux/macOS + Windows defaults)
        env_candidates = [
            os.environ.get("HUGGINGFACE_HUB_CACHE"),
            os.environ.get("HF_HOME"),
            os.environ.get("HF_HUB_CACHE"),
            self.cache_dir,
        ]
        win_local = os.path.join(os.environ.get("LOCALAPPDATA", ""), "huggingface")
        win_roam = os.path.join(os.environ.get("APPDATA", ""), "huggingface")
        base_candidates = [
            p for p in (
                *(env_candidates or []),
                os.path.expanduser("~/.cache/huggingface"),
                win_local,
                win_roam,
            )
            if p
        ]
        # Search both base and base/hub for each candidate
        candidates = []
        for b in base_candidates:
            pb = Path(b)
            candidates.append(pb)
            candidates.append(pb / "hub")
        results: List[tuple[str, Path]] = []
        scanned_under: List[str] = []
        for base in candidates:
            if not base.exists():
                continue
            scanned_under.append(str(base))
            for models_dir in base.glob("models--*--*"):
                parts = models_dir.name.split("--")
                if len(parts) < 3:
                    continue
                org = parts[1]
                repo = parts[2]
                repo_id = f"{org}/{repo}"
                snap_root = models_dir / "snapshots"
                if not snap_root.exists():
                    continue
                snapshots = sorted(
                    [p for p in snap_root.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not snapshots:
                    continue
                results.append((repo_id, snapshots[0]))
        self._logger.warning(
            f"HFCacheScanner: scanned {len(results)} model snapshots under {', '.join(scanned_under) or 'N/A'}")
        return results

    def scan_flux_models(self) -> Dict[str, List[Dict[str, Any]]]:
        transformers: List[Dict[str, Any]] = []
        clip_encoders: List[Dict[str, Any]] = []
        t5_encoders: List[Dict[str, Any]] = []
        for repo_id, snap_path in self._iter_model_snapshots():
            rid_lower = repo_id.lower()
            entry = {
                "display_name": repo_id,
                "path": snap_path,
                "repo_id": repo_id,
            }
            if "flux" in rid_lower:
                transformers.append(entry)
            # Many FLUX repos use Google's SigLIP text encoder; treat it as CLIP-compatible
            elif ("clip" in rid_lower) or ("siglip" in rid_lower):
                clip_encoders.append(entry)
            elif "t5" in rid_lower:
                t5_encoders.append(entry)
        # Helpful breakdown for troubleshooting discovery issues
        try:
            self._logger.warning(
                f"HFCacheScanner: categorized transformers={len(transformers)}, clip/siglip={len(clip_encoders)}, t5={len(t5_encoders)}"
            )
        except Exception:
            pass
        return {
            "transformers": transformers,
            "clip_encoders": clip_encoders,
            "t5_encoders": t5_encoders,
        }


