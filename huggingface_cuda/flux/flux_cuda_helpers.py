from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from griptape.artifacts import ImageUrlArtifact

# Import with fallback for standalone loading
try:
    from .flux_config import FluxConfig
except ImportError:
    import importlib.util
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("flux_config", os.path.join(current_dir, "flux_config.py"))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    FluxConfig = config_module.FluxConfig

class SeedManager:
    @staticmethod
    def resolve(seed_in: int, mode: str, last_seed: int) -> int:
        if mode == "fixed":
            return seed_in
        if mode == "increment":
            return last_seed + 1
        if mode == "decrement":
            return max(last_seed - 1, 0)
        if mode == "randomize":
            import random
            return random.randint(0, 2**32 - 1)
        return seed_in
    
class PromptBuilder:
    """Utility for merging main + extra prompts with basic cleanup."""

    @staticmethod
    def combine(main_prompt: str | None, extra_prompts: List[str] | None, *, limit: int = 5) -> Tuple[str, List[str]]:
        main = (main_prompt or "").strip()
        prompts: List[str] = [main] if main else []
        if extra_prompts:
            prompts.extend(p.strip() for p in extra_prompts if p and str(p).strip())
        prompts = [p for p in prompts if p][:limit]
        if not prompts:
            combined = ""
        elif len(prompts) == 1:
            combined = prompts[0]
        else:
            combined = ", ".join(prompts)
        return combined, prompts


class ImageInputManager:
    """Handles dynamic control_images ParameterList injection."""

    def __init__(self, node, cfg: FluxConfig):
        from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
        self.node = node
        self.cfg = cfg
        # template ParameterList – cloned when inserted
        self._template = ParameterList(
            name="control_images",
            input_types=["ImageUrlArtifact", "list[ImageUrlArtifact]"],
            default_value=[],
            tooltip="Optional control / conditioning images.",
            allowed_modes={ParameterMode.INPUT},
            ui_options={"display_name": "Control Images"},
        )

    # ------------------------------------------------------------------
    def sync(self, model_id: str) -> None:
        supports = self.cfg.supports_image_input(model_id)
        present = self.node.get_parameter_by_name("control_images") is not None
        if supports and not present:
            self.node.add_parameter(self._template)
        elif not supports and present:
            param = self.node.get_parameter_by_name("control_images")
            if param:
                self.node.remove_element(param)

    # ------------------------------------------------------------------
    def get_images(self, model_id: str):
        if not self.cfg.supports_image_input(model_id):
            return []
        max_n = self.cfg.max_control_images(model_id)
        try:
            images = self.node.get_parameter_list_value("control_images")
        except Exception:
            images = []
        # keep only ImageUrlArtifact instances and cap
        images = [img for img in images if isinstance(img, ImageUrlArtifact)][:max_n]
        return images


# ----------------------------------------------------------------------

def create_error_image(tmp_dir: Path, msg: str, exc: Exception) -> str:
    """Create a tiny PNG with the error text and return file:// URL."""
    from PIL import Image, ImageDraw
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (600, 120), color=(30, 30, 30))
    d = ImageDraw.Draw(img)
    d.text((10, 10), msg[:200], fill=(255, 0, 0))
    filepath = tmp_dir / f"flux_error_{int(__import__('time').time()*1000)}.png"
    img.save(filepath)
    return f"file://{filepath}"


# Parameter validation utilities
class ParameterValidator:
    """Parameter validation utilities for FLUX models"""
    
    @staticmethod
    def validate_dimensions(width: int, height: int) -> tuple[int, int]:
        """Validate and clamp image dimensions to FLUX-compatible ranges"""
        # FLUX works best with dimensions that are multiples of 8
        width = max(256, min(2048, width))
        height = max(256, min(2048, height))
        
        # Round to nearest multiple of 8
        width = (width + 7) // 8 * 8
        height = (height + 7) // 8 * 8
        
        return width, height
    
    @staticmethod
    def validate_steps(steps: int, model_id: str) -> int:
        """Validate inference steps based on model type"""
        if "schnell" in model_id.lower():
            # FLUX.1-schnell works well with 1-8 steps
            return max(1, min(8, steps))
        else:
            # FLUX.1-dev works well with 15-50 steps
            return max(1, min(50, steps))
    
    @staticmethod
    def validate_guidance_scale(guidance: float, model_id: str) -> float:
        """Validate guidance scale based on model type"""
        if "schnell" in model_id.lower():
            # FLUX.1-schnell typically uses guidance_scale=0
            return 0.0
        else:
            # FLUX.1-dev typically uses guidance_scale 1-10
            return max(0.0, min(10.0, guidance))
    
    @staticmethod
    def validate_seed(seed: int) -> int:
        """Validate seed value"""
        if seed == -1:
            import random
            return random.randint(0, 2**32 - 1)
        return max(0, min(2**32 - 1, seed))


# Error handling utilities
class FluxErrorHandler:
    """Error handling utilities for FLUX operations"""
    
    @staticmethod
    def handle_memory_error(error: Exception, model_id: str) -> str:
        """Generate helpful error message for memory issues"""
        error_str = str(error).lower()
        
        if "out of memory" in error_str or "cuda out of memory" in error_str:
            return (
                f"GPU ran out of memory loading {model_id}. "
                f"Try: 1) Use quantization (8-bit/4-bit), 2) Reduce image size, "
                f"3) Close other GPU applications, 4) Restart the workflow"
            )
        elif "no kernel image" in error_str:
            return (
                f"GPU kernels not available for {model_id}. "
                f"This typically happens with newer GPUs. Try: 1) Update CUDA drivers, "
                f"2) Use CPU inference, 3) Try a different quantization setting"
            )
        else:
            return f"Memory/GPU error with {model_id}: {error}"
    
    @staticmethod
    def handle_model_error(error: Exception, model_id: str) -> str:
        """Generate helpful error message for model loading issues"""
        error_str = str(error).lower()
        
        if "not found" in error_str or "does not exist" in error_str:
            return (
                f"Model {model_id} not found. "
                f"Try: 1) Check model ID spelling, 2) Ensure model is cached, "
                f"3) Check internet connection for download"
            )
        elif "permission" in error_str or "access" in error_str:
            return (
                f"Permission error accessing {model_id}. "
                f"Try: 1) Check HuggingFace token, 2) Verify model access rights, "
                f"3) Re-authenticate with HuggingFace"
            )
        else:
            return f"Model loading error with {model_id}: {error}"
    
    @staticmethod
    def categorize_error(error: Exception) -> str:
        """Categorize error type for better handling"""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ["memory", "cuda", "gpu"]):
            return "memory"
        elif any(keyword in error_str for keyword in ["not found", "missing", "exists"]):
            return "model"
        elif any(keyword in error_str for keyword in ["permission", "access", "token"]):
            return "auth"
        elif any(keyword in error_str for keyword in ["network", "connection", "timeout"]):
            return "network"
        else:
            return "unknown"


# Performance utilities
class PerformanceUtils:
    """Performance optimization utilities"""
    
    @staticmethod
    def estimate_memory_usage(width: int, height: int, quantization: str = "none") -> dict:
        """Estimate GPU memory usage for given parameters"""
        # Base memory for FLUX model (approximate)
        base_memory_gb = {
            "none": 12.0,      # Full precision
            "8-bit": 6.5,      # 8-bit quantization  
            "4-bit": 4.0,      # 4-bit quantization
        }.get(quantization, 12.0)
        
        # Additional memory for image generation (latents, activations)
        pixels = width * height
        activation_memory_gb = pixels * 16 * 4 / (1024**3)  # Rough estimate
        
        total_memory_gb = base_memory_gb + activation_memory_gb
        
        return {
            "base_model_gb": base_memory_gb,
            "activation_gb": activation_memory_gb,
            "total_estimated_gb": total_memory_gb,
            "quantization": quantization,
            "resolution": f"{width}x{height}"
        }
    
    @staticmethod
    def suggest_optimization(available_memory_gb: float, required_memory_gb: float) -> list[str]:
        """Suggest optimizations based on memory constraints"""
        suggestions = []
        
        if required_memory_gb > available_memory_gb:
            deficit = required_memory_gb - available_memory_gb
            suggestions.append(f"⚠️ Memory deficit: {deficit:.1f}GB")
            
            if deficit > 6:
                suggestions.append("💡 Try 4-bit quantization (saves ~8GB)")
            elif deficit > 3:
                suggestions.append("💡 Try 8-bit quantization (saves ~5GB)")
            
            suggestions.append("💡 Reduce image resolution")
            suggestions.append("💡 Close other GPU applications")
            suggestions.append("💡 Consider CPU inference for large models")
        else:
            suggestions.append(f"✅ Sufficient memory: {available_memory_gb:.1f}GB available")
            
        return suggestions


# Shared utility functions
def format_memory_size(bytes_size: int) -> str:
    """Format memory size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


def safe_import(module_name: str, fallback_value=None):
    """Safely import a module with fallback"""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        return fallback_value


def log_performance_metrics(operation: str, duration: float, memory_used: int = None):
    """Log performance metrics in a standardized format"""
    message = f"[FLUX PERF] {operation}: {duration:.2f}s"
    if memory_used:
        message += f", {format_memory_size(memory_used)} memory"
    print(message)