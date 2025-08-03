"""
FLUX-specific implementations for CUDA
"""

# Export main flux inference node
# Import with fallback for standalone loading
try:
    from .flux_inference import FluxInference
except ImportError:
    import importlib.util
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("flux_inference", os.path.join(current_dir, "flux_inference.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    FluxInference = module.FluxInference

__all__ = [
    "FluxInference"
] 