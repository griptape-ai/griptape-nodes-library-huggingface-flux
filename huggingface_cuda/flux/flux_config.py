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

    # ------------------------------------------------------------------
    # Device Management Settings
    # ------------------------------------------------------------------
    def get_device_strategy(self) -> str:
        """Get device placement strategy: 'auto', 'cpu_offload', or 'manual'"""
        return self._cfg.get("device_management", {}).get("strategy", "cpu_offload")
    
    def should_enable_cpu_offload(self) -> bool:
        """Whether to enable CPU offloading"""
        return bool(self._cfg.get("device_management", {}).get("enable_cpu_offload", True))
    
    def should_enable_sequential_cpu_offload(self) -> bool:
        """Whether to enable sequential CPU offloading (faster than full offload)"""
        return bool(self._cfg.get("device_management", {}).get("enable_sequential_cpu_offload", False))
    
    def get_manual_device_map(self) -> Dict[str, Any]:
        """Get manual device map (only used if strategy is 'manual')"""
        return self._cfg.get("device_management", {}).get("manual_device_map", {})

    # ------------------------------------------------------------------
    # Memory Management Settings  
    # ------------------------------------------------------------------
    def get_pytorch_allocator_config(self) -> Dict[str, Any]:
        """Get PyTorch CUDA allocator configuration"""
        return self._cfg.get("memory_management", {}).get("pytorch_allocator", {
            "garbage_collection_threshold": 0.8,
            "max_split_size_mb": 128,
            "expandable_segments": True
        })
    
    def get_quantization_memory_limits(self, quantization: str) -> Dict[str, int]:
        """Get memory limits for specific quantization type"""
        limits = self._cfg.get("memory_management", {}).get("quantization_memory_limits", {})
        return limits.get(quantization, limits.get("8-bit", {
            "gpu_memory_gb": 10,
            "cpu_memory_gb": 8
        }))
    
    def get_memory_cleanup_config(self) -> Dict[str, bool]:
        """Get aggressive memory cleanup configuration"""
        return self._cfg.get("memory_management", {}).get("aggressive_memory_cleanup", {
            "enabled": True,
            "pre_inference_cleanup": True,
            "emergency_vae_cpu_fallback": True
        })
