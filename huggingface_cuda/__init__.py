"""
HuggingFace CUDA Nodes Library

CUDA-optimized nodes for HuggingFace models with GPU selection and memory management.
"""

# Import backend functions from the advanced library
from .library_loader import get_shared_backend, initialize_cuda_backend

__version__ = "0.1.0" 