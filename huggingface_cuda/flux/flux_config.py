import json
from pathlib import Path
from typing import Any, Dict


class FluxConfig:
    """Simple configuration loader for CUDA FLUX node.

    Mirrors the MLX variant but only exposes the pieces the CUDA node needs
    right now (image-input capabilities).
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).with_name("flux_config.json")
        self._path = Path(config_path)
        if not self._path.exists():
            raise FileNotFoundError(f"flux_config.json not found at {self._path}")
        with open(self._path, "r", encoding="utf-8") as f:
            self._cfg = json.load(f)

    # ------------------------------------------------------------------
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        return self._cfg.get("models", {}).get(model_id, {})

    # ------------------------------------------------------------------
    def supports_image_input(self, model_id: str) -> bool:
        return bool(self.get_model_config(model_id).get("supports_image_input", False))

    # ------------------------------------------------------------------
    def max_control_images(self, model_id: str) -> int:
        model_cfg = self.get_model_config(model_id)
        if "max_control_images" in model_cfg:
            return int(model_cfg["max_control_images"])
        return int(self._cfg.get("global_defaults", {}).get("default_image_input_limit", 1))
