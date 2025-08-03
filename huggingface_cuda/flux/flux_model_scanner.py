"""
Model discovery and validation utilities for FLUX models.

Handles scanning HuggingFace cache for available FLUX models and validating their structure.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# Supported Flux models with their configurations
FLUX_MODELS = {
    "black-forest-labs/FLUX.1-dev": {
        "display_name": "FLUX.1 Dev",
        "default_steps": 20,
        "max_steps": 50,
        "supports_guidance": True,
        "default_guidance": 7.5
    },
    "black-forest-labs/FLUX.1-schnell": {
        "display_name": "FLUX.1 Schnell", 
        "default_steps": 4,
        "max_steps": 8,
        "supports_guidance": False,
        "default_guidance": 1.0
    }
}


class FluxModelScanner:
    """Scans and validates FLUX models in HuggingFace cache."""
    
    def __init__(self):
        """Initialize the model scanner."""
        pass
    
    def scan_available_models(self) -> List[str]:
        """Dynamically scan HuggingFace cache for FLUX models by analyzing model structure."""
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            # If HF not available, return config models as fallback
            available = list(FLUX_MODELS.keys())
            print(f"[FLUX SCAN] HuggingFace hub not available, using defaults: {available}")
            return available
        
        available = []
        try:
            cache_info = scan_cache_dir()
            
            # Handle different HF hub API versions
            cache_dir = getattr(cache_info, 'cache_dir', getattr(cache_info, 'cache_path', 'unknown'))
            print(f"[FLUX SCAN] Scanning cache directory: {cache_dir}")
            
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                
                # First check if it has FLUX in the name at all
                if not self.is_flux_model(repo_id):
                    continue
                
                # Now analyze the actual model structure
                if len(repo.revisions) > 0:
                    # Get the latest revision's snapshot path
                    latest_revision = next(iter(repo.revisions))
                    snapshot_path = Path(latest_revision.snapshot_path)
                    
                    if self.is_base_flux_model(snapshot_path, repo_id):
                        # Critical: Check if model is actually loadable
                        if self.is_model_loadable(repo_id):
                            available.append(repo_id)
                            print(f"[FLUX SCAN] ✅ Found complete FLUX model: {repo_id}")
                        else:
                            print(f"[FLUX SCAN] ❌ Skipping incomplete FLUX model: {repo_id}")
                    else:
                        print(f"[FLUX SCAN] Skipping specialized FLUX model: {repo_id}")
                        
        except Exception as e:
            print(f"[FLUX SCAN] Error scanning cache: {e}")
            # If scanning fails, return config models as fallback
            available = list(FLUX_MODELS.keys())
            
        # Always return at least one option (fallback to config models)
        if not available:
            available = list(FLUX_MODELS.keys())
            print("[FLUX SCAN] No models found in cache, using config defaults")
        
        print(f"[FLUX SCAN] Available models: {available}")
        return available

    def is_flux_model(self, model_id: str) -> bool:
        """Check if model_id appears to be a FLUX model based on patterns"""
        model_lower = model_id.lower()
        
        # Exclude encoder-only repositories
        encoder_exclusions = [
            "text_encoders" in model_lower,
            "clip_encoders" in model_lower, 
            "t5_encoders" in model_lower,
            model_lower.endswith("/clip"),
            model_lower.endswith("/t5"),
            "encoder" in model_lower and ("clip" in model_lower or "t5" in model_lower)
        ]
        
        if any(encoder_exclusions):
            return False
        
        # Exclude LoRA repositories
        lora_exclusions = [
            "lora" in model_lower,
            "-lora-" in model_lower,
            "lora_" in model_lower,
            model_lower.endswith("_lora"),
            model_lower.endswith("-lora"),
            "adapter" in model_lower and "flux" in model_lower
        ]
        
        if any(lora_exclusions):
            return False
        
        # Check for FLUX patterns
        flux_patterns = [
            "flux" in model_lower,
            "black-forest" in model_lower and "flux" in model_lower,
        ]
        
        return any(flux_patterns)

    def is_base_flux_model(self, snapshot_path: Path, repo_id: str) -> bool:
        """Analyze model structure to determine if it's a base FLUX text-to-image model."""
        try:
            # First check: exclude encoder-only repositories by structure
            if self.is_encoder_only_repository(snapshot_path, repo_id):
                print(f"[FLUX SCAN] {repo_id}: Encoder-only repository detected, skipping")
                return False
            
            # Check for model_index.json (indicates diffusers pipeline)
            model_index_path = snapshot_path / "model_index.json"
            if model_index_path.exists():
                with open(model_index_path) as f:
                    model_index = json.load(f)
                
                # Check if it's a FLUX pipeline with the right components
                if model_index.get("_class_name") in ["FluxPipeline", "FlowMatchEulerDiscreteScheduler"]:
                    # Make sure it has the core components for text-to-image
                    required_components = ["transformer", "scheduler", "vae"]
                    has_required = all(comp in model_index for comp in required_components)
                    
                    # Make sure it's NOT a ControlNet (they have controlnet in components)
                    is_controlnet = "controlnet" in model_index
                    
                    if has_required and not is_controlnet:
                        print(f"[FLUX SCAN] {repo_id}: Valid FLUX pipeline")
                        return True
                    elif is_controlnet:
                        print(f"[FLUX SCAN] {repo_id}: ControlNet detected, skipping")
                        return False
            
            # Check individual model config files
            config_path = snapshot_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                
                # Check if it's a ControlNet by architecture
                if config.get("_class_name") == "FluxControlNetModel":
                    print(f"[FLUX SCAN] {repo_id}: ControlNet config detected, skipping")
                    return False
                
                # Check if it's a transformer model (the main FLUX component)
                if config.get("_class_name") == "FluxTransformer2DModel":
                    print(f"[FLUX SCAN] {repo_id}: FLUX transformer detected")
                    return True
            
            # Fallback: check for common ControlNet indicators in the directory structure
            controlnet_indicators = [
                "controlnet",
                "control_net", 
                "canny",
                "depth",
                "pose",
                "inpaint",
                "fill"
            ]
            
            repo_lower = repo_id.lower()
            if any(indicator in repo_lower for indicator in controlnet_indicators):
                print(f"[FLUX SCAN] {repo_id}: ControlNet/specialized model detected by name patterns")
                return False
            
            # If we can't determine definitively, default to including it
            print(f"[FLUX SCAN] {repo_id}: Assuming base model (no clear specialization detected)")
            return True
            
        except Exception as e:
            print(f"[FLUX SCAN] Error analyzing {repo_id}: {e}")
            # When in doubt, include it
            return True

    def is_encoder_only_repository(self, snapshot_path: Path, repo_id: str) -> bool:
        """Check if this repository only contains text encoders (T5/CLIP)"""
        try:
            # List all files in the repository
            all_files = list(snapshot_path.rglob("*.safetensors"))
            
            # Check if it only contains encoder files and no core FLUX components
            has_transformer = any("transformer" in str(f) for f in all_files)
            has_vae = any("vae" in str(f) for f in all_files)
            has_scheduler = (snapshot_path / "scheduler").exists()
            
            # If it has no core FLUX components, it's likely an encoder-only repository
            if not (has_transformer or has_vae or has_scheduler):
                # Double-check by looking for encoder patterns
                encoder_files = [f for f in all_files if any(pattern in str(f).lower() for pattern in ["clip", "t5", "text_encoder"])]
                if encoder_files and len(encoder_files) == len(all_files):
                    return True
            
            return False
            
        except Exception as e:
            print(f"[FLUX SCAN] Error checking encoder-only status for {repo_id}: {e}")
            return False

    def is_model_loadable(self, model_id: str) -> bool:
        """Test if a model can actually be loaded without errors."""
        try:
            # Get shared backend modules
            from huggingface_hub import snapshot_download
            
            # First check: Try to get model info without downloading
            try:
                # Use offline mode to check if model is already fully cached
                snapshot_path = snapshot_download(
                    repo_id=model_id,
                    local_files_only=True,  # Only use local cache
                    allow_patterns=["config.json", "model_index.json", "*.safetensors"]
                )
                
                # Check if essential files exist
                snapshot_path = Path(snapshot_path)
                
                # For FLUX models, we need either model_index.json (pipeline) or config.json (transformer)
                has_model_index = (snapshot_path / "model_index.json").exists()
                has_config = (snapshot_path / "config.json").exists()
                
                if not (has_model_index or has_config):
                    print(f"[FLUX SCAN] {model_id}: Missing essential config files")
                    return False
                
                # Check for model weight files
                safetensors_files = list(snapshot_path.rglob("*.safetensors"))
                if not safetensors_files:
                    print(f"[FLUX SCAN] {model_id}: No model weight files found")
                    return False
                
                # Quick validation: try to load config without actually loading weights
                if has_model_index:
                    with open(snapshot_path / "model_index.json") as f:
                        model_index = json.load(f)
                    if not model_index.get("_class_name"):
                        print(f"[FLUX SCAN] {model_id}: Invalid model_index.json")
                        return False
                
                print(f"[FLUX SCAN] {model_id}: Validation passed - model is loadable")
                return True
                
            except Exception as e:
                print(f"[FLUX SCAN] {model_id}: Not fully cached or corrupted - {e}")
                return False
                
        except Exception as e:
            print(f"[FLUX SCAN] {model_id}: Error during loadability check - {e}")
            return False