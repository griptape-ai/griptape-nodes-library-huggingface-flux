import json
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Robust import for gpu_utils with fallback handling
def get_gpu_utils_functions():
    """Get GPU utility functions with robust import handling"""
    try:
        # Try relative import first
        from .gpu_utils import get_available_gpus, get_gpu_memory_info
        return get_available_gpus, get_gpu_memory_info
    except ImportError:
        try:
            # Try absolute import
            from huggingface_cuda.shared.gpu_utils import get_available_gpus, get_gpu_memory_info
            return get_available_gpus, get_gpu_memory_info
        except ImportError:
            try:
                # Try direct module loading
                current_dir = os.path.dirname(os.path.abspath(__file__))
                utils_path = os.path.join(current_dir, "gpu_utils.py")
                if os.path.exists(utils_path):
                    spec = importlib.util.spec_from_file_location("gpu_utils", utils_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module.get_available_gpus, module.get_gpu_memory_info
            except Exception as e:
                print(f"[GPU CONFIG] Error loading GPU utils: {e}")
                # Return fallback functions
                def fallback_get_available_gpus():
                    return []
                def fallback_get_gpu_memory_info(device_id):
                    return {}
                return fallback_get_available_gpus, fallback_get_gpu_memory_info

# Load the functions at module level
get_available_gpus, get_gpu_memory_info = get_gpu_utils_functions()


class GPUConfig:
    """GPU configuration manager for HuggingFace CUDA library"""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "gpu_config.json"
        
        with open(config_path) as f:
            self.config = json.load(f)
    
    def get_quantization_defaults(self, quantization: str) -> Dict[str, Any]:
        """Get default configuration for a quantization type"""
        defaults = self.config["quantization_defaults"].get(quantization, {})
        
        # Add global defaults
        result = {
            "auto_detect_gpu": self.config["auto_detect_gpu"],
            "fallback_to_cpu": self.config["fallback_to_cpu"],
            **defaults
        }
        
        return result
    
    def calculate_memory_limits(self, gpu_device: int, gpu_memory_fraction: float, 
                              cpu_memory_gb: float) -> Dict[str, Any]:
        """Calculate actual memory limits based on system and fractions"""
        try:
            # Get GPU info
            available_gpus = get_available_gpus()
            gpu_info = None
            for gpu in available_gpus:
                if gpu['id'] == gpu_device:
                    gpu_info = gpu
                    break
            
            if not gpu_info:
                raise ValueError(f"GPU device {gpu_device} not found")
            
            # Calculate GPU memory limit
            total_gpu_memory = gpu_info['memory_gb']
            buffer_gb = self.config["memory_calculation"]["gpu_memory_buffer_gb"]
            usable_gpu_memory = max(
                total_gpu_memory * gpu_memory_fraction - buffer_gb,
                self.config["memory_calculation"]["min_gpu_memory_gb"]
            )
            
            # Build memory configuration
            memory_config = {
                gpu_device: f"{usable_gpu_memory:.1f}GB",
                "cpu": f"{cpu_memory_gb:.1f}GB"
            }
            
            return {
                "max_memory": memory_config,
                "gpu_total_gb": total_gpu_memory,
                "gpu_allocated_gb": usable_gpu_memory,
                "cpu_allocated_gb": cpu_memory_gb,
                "memory_fraction_used": gpu_memory_fraction
            }
            
        except Exception as e:
            print(f"[GPU CONFIG] Error calculating memory limits: {e}")
            # Fallback to conservative defaults
            return {
                "max_memory": {gpu_device: "12GB", "cpu": f"{cpu_memory_gb:.1f}GB"},
                "gpu_total_gb": 16,  # Conservative guess
                "gpu_allocated_gb": 12,
                "cpu_allocated_gb": cpu_memory_gb,
                "memory_fraction_used": gpu_memory_fraction
            }
    
    def get_available_gpu_choices(self) -> List[Tuple[str, int]]:
        """Get formatted choices for GPU dropdown"""
        choices = []
        try:
            available_gpus = get_available_gpus()
            
            for gpu in available_gpus:
                if gpu['is_available']:
                    # Format: "RTX 4090 (24GB) - GPU 0"
                    name = f"{gpu['name']} ({gpu['memory_gb']:.0f}GB) - GPU {gpu['id']}"
                    choices.append((name, gpu['id']))
                    
            # Add auto-detect option
            if choices:
                choices.insert(0, ("Auto-detect optimal GPU", -1))
            else:
                choices = [("No CUDA GPUs available", -1)]
                
        except Exception as e:
            print(f"[GPU CONFIG] Error getting GPU choices: {e}")
            choices = [("GPU detection failed", -1)]
        
        return choices
    
    def get_system_profile(self, gpu_name: str) -> Optional[Dict[str, Any]]:
        """Get system profile based on GPU name"""
        gpu_lower = gpu_name.lower()
        
        for profile_name, profile in self.config["system_profiles"].items():
            if profile_name.replace("_", " ") in gpu_lower:
                return profile
        
        return None
    
    def get_recommended_settings(self, gpu_device: int) -> Dict[str, Any]:
        """Get recommended settings for a specific GPU"""
        try:
            available_gpus = get_available_gpus()
            gpu_info = None
            for gpu in available_gpus:
                if gpu['id'] == gpu_device:
                    gpu_info = gpu
                    break
            
            if not gpu_info:
                return self.get_quantization_defaults("4-bit")
            
            # Check for system profile
            profile = self.get_system_profile(gpu_info['name'])
            if profile:
                defaults = self.get_quantization_defaults(profile['optimal_quantization'])
                defaults['gpu_memory_fraction'] = profile['recommended_fraction']
                defaults['recommended_quantization'] = profile['optimal_quantization']
                return defaults
            
            # Fallback based on GPU memory
            if gpu_info['memory_gb'] >= 20:
                return self.get_quantization_defaults("4-bit")
            elif gpu_info['memory_gb'] >= 12:
                return self.get_quantization_defaults("8-bit")
            else:
                defaults = self.get_quantization_defaults("8-bit")
                defaults['gpu_memory_fraction'] = 0.6  # More conservative
                return defaults
                
        except Exception as e:
            print(f"[GPU CONFIG] Error getting recommendations: {e}")
            return self.get_quantization_defaults("4-bit")
    
    def validate_settings(self, gpu_memory_fraction: float, cpu_memory_gb: float) -> Dict[str, Any]:
        """Validate settings and return any warnings or errors"""
        validation = self.config["validation"]
        issues = []
        
        # Check GPU memory fraction
        if gpu_memory_fraction < validation["min_memory_fraction"]:
            issues.append(f"GPU memory fraction too low (min: {validation['min_memory_fraction']})")
        elif gpu_memory_fraction > validation["max_memory_fraction"]:
            issues.append(f"GPU memory fraction too high (max: {validation['max_memory_fraction']})")
        
        # Check CPU memory
        if cpu_memory_gb < validation["min_cpu_memory_gb"]:
            issues.append(f"CPU memory too low (min: {validation['min_cpu_memory_gb']}GB)")
        elif cpu_memory_gb > validation["max_cpu_memory_gb"]:
            issues.append(f"CPU memory too high (max: {validation['max_cpu_memory_gb']}GB)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": []
        }
    
    def build_system_constraints_dict(self, gpu_device: int, gpu_memory_fraction: float,
                                    cpu_memory_gb: float, device_map: str = "auto") -> Dict[str, Any]:
        """Build the complete system constraints dictionary for the FLUX node"""
        
        # Calculate memory limits
        memory_info = self.calculate_memory_limits(gpu_device, gpu_memory_fraction, cpu_memory_gb)
        
        # Validate settings
        validation = self.validate_settings(gpu_memory_fraction, cpu_memory_gb)
        
        # Build the complete configuration
        constraints = {
            "gpu_device": gpu_device,
            "gpu_memory_fraction": gpu_memory_fraction,
            "cpu_memory_gb": cpu_memory_gb,
            "device_map": device_map,
            "max_memory": memory_info["max_memory"],
            "low_cpu_mem_usage": cpu_memory_gb > 16,  # Enable for systems with enough RAM
            
            # Metadata for debugging/display
            "metadata": {
                "gpu_total_gb": memory_info["gpu_total_gb"],
                "gpu_allocated_gb": memory_info["gpu_allocated_gb"],
                "cpu_allocated_gb": memory_info["cpu_allocated_gb"],
                "memory_fraction_used": memory_info["memory_fraction_used"],
                "validation": validation
            }
        }
        
        return constraints 