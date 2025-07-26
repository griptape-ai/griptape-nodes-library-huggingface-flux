from typing import Any
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

try:
    from .shared.gpu_utils import get_available_gpus
    print("[GPU CONFIG DEBUG] Successfully imported get_available_gpus from shared.gpu_utils")
except ImportError as e:
    print(f"[GPU CONFIG DEBUG] Relative import failed: {e}")
    try:
        # Try absolute import
        from huggingface_cuda.shared.gpu_utils import get_available_gpus
        print("[GPU CONFIG DEBUG] Successfully imported get_available_gpus via absolute import")
    except ImportError as e2:
        print(f"[GPU CONFIG DEBUG] Absolute import failed: {e2}")
        try:
            # Try direct module loading
            import os
            import importlib.util
            current_dir = os.path.dirname(os.path.abspath(__file__))
            gpu_utils_path = os.path.join(current_dir, "shared", "gpu_utils.py")
            print(f"[GPU CONFIG DEBUG] Trying to load from: {gpu_utils_path}")
            
            spec = importlib.util.spec_from_file_location("gpu_utils", gpu_utils_path)
            gpu_utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gpu_utils_module)
            get_available_gpus = gpu_utils_module.get_available_gpus
            print("[GPU CONFIG DEBUG] Successfully loaded get_available_gpus via direct module loading")
        except Exception as e3:
            print(f"[GPU CONFIG DEBUG] All import methods failed: {e3}, using fallback")
            def get_available_gpus():
                return [{'id': 0, 'name': 'Fallback GPU', 'memory_gb': 8.0, 'is_available': True}]


class GPUConfiguration(ControlNode):
    """Simple GPU Configuration for FLUX inference"""
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Setup"
        self.description = "Configure GPU and memory settings for FLUX inference"
        
        # Get available GPUs and create simple choices (like model selection)
        available_gpus = get_available_gpus()
        print(f"[GPU CONFIG DEBUG] Available GPUs: {available_gpus}")
        gpu_choices = []
        
        # Add auto-detect option first
        gpu_choices.append("auto")
        
        # Add each available GPU by name
        for gpu in available_gpus:
            if gpu['is_available']:
                gpu_name = f"{gpu['name']} ({gpu['memory_gb']:.0f}GB)"
                gpu_choices.append(gpu_name)
        
        # Default to auto
        default_gpu = gpu_choices[0] if gpu_choices else "auto"
        
        # Parameters - using same pattern as FLUX model selection
        self.add_parameter(Parameter(
            name="gpu_device",
            tooltip="Select which GPU to use for inference",
            type="str",
            default_value=default_gpu,
            allowed_modes={ParameterMode.PROPERTY},
            traits={Options(choices=gpu_choices)}
        ))
        
        self.add_parameter(Parameter(
            name="gpu_memory_fraction",
            tooltip="Percentage of GPU memory to allocate (0.1 = 10%, 0.9 = 90%)",
            type="float",
            default_value=0.8,
            allowed_modes={ParameterMode.PROPERTY},
            traits={Slider(min_val=0.1, max_val=0.95)}
        ))
        
        self.add_parameter(Parameter(
            name="cpu_memory_gb",
            tooltip="System RAM (GB) for FLUX CPU offload. Lower values leave more memory for video editing, higher values allow better FLUX performance. Recommended: 16GB (video editing), 20GB (FLUX only), 24GB (heavy workloads).",
            type="float",
            default_value=16.0,
            allowed_modes={ParameterMode.PROPERTY},
            traits={Slider(min_val=8.0, max_val=32.0)}
        ))
        
        self.add_parameter(Parameter(
            name="allow_low_memory",
            tooltip="⚠️ UNSAFE: Override memory safety checks. Use only if you understand the risks of potential system hangs.",
            type="bool",
            default_value=False,
            allowed_modes={ParameterMode.PROPERTY}
        ))
        
        self.add_parameter(Parameter(
            name="system_constraints",
            output_type="dict",
            tooltip="GPU configuration for FLUX nodes",
            allowed_modes={ParameterMode.OUTPUT}
        ))

    def process(self) -> None:
        """Generate simple GPU configuration"""
        # Get parameter values (now simple strings like model selection)
        gpu_device_value = self.get_parameter_value("gpu_device")
        gpu_memory_fraction = self.get_parameter_value("gpu_memory_fraction")
        cpu_memory_gb = self.get_parameter_value("cpu_memory_gb")
        allow_low_memory = self.get_parameter_value("allow_low_memory")
        
        # Get available GPUs for device selection and memory calculation
        available_gpus = get_available_gpus()
        
        # Convert GPU selection to device config
        if gpu_device_value == "auto":
            gpu_device = "auto"
        else:
            # Extract GPU ID from the display name (find the GPU that matches)
            gpu_device = 0  # Default fallback
            for gpu in available_gpus:
                gpu_name = f"{gpu['name']} ({gpu['memory_gb']:.0f}GB)"
                if gpu_name == gpu_device_value:
                    gpu_device = gpu['id']
                    break
        
        # Calculate memory limits for PyTorch/Accelerate
        # Get actual GPU memory for the selected device
        selected_gpu_memory_gb = 16.0  # Default fallback
        for gpu in available_gpus:
            if gpu['id'] == gpu_device:
                selected_gpu_memory_gb = gpu['memory_gb']
                break
        
        # Calculate actual memory limits
        gpu_memory_limit_gb = selected_gpu_memory_gb * gpu_memory_fraction
        
        # Create configuration with memory limits
        config = {
            "gpu_device": gpu_device,
            "gpu_memory_fraction": float(gpu_memory_fraction),
            "cpu_memory_gb": float(cpu_memory_gb),
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "max_memory": {
                gpu_device: f"{gpu_memory_limit_gb:.0f}GB",
                "cpu": f"{cpu_memory_gb:.0f}GB"
            }
        }
        
        # Add allow_low_memory if enabled
        if allow_low_memory:
            config["allow_low_memory"] = True
            print(f"[GPU CONFIG] ⚠️ Low memory override enabled - bypassing safety checks")
        
        self.parameter_output_values["system_constraints"] = config
        
        # Log the configuration with memory limits and usage recommendations
        print(f"[GPU CONFIG] Configuration: {gpu_device_value}, {gpu_memory_fraction*100:.0f}% GPU memory, {cpu_memory_gb}GB CPU memory")
        print(f"[GPU CONFIG] Memory limits: GPU {gpu_device} = {gpu_memory_limit_gb:.0f}GB, CPU = {cpu_memory_gb:.0f}GB")
        
        # Show memory strategy info
        if cpu_memory_gb <= 12:
            print(f"[GPU CONFIG] 💡 Conservative CPU memory - ideal for heavy multitasking/video editing")
        elif cpu_memory_gb <= 18:
            print(f"[GPU CONFIG] 💡 Balanced CPU memory - good for video editing + FLUX workflows")
        elif cpu_memory_gb <= 24:
            print(f"[GPU CONFIG] 💡 FLUX-optimized CPU memory - best for dedicated FLUX workloads")
        else:
            print(f"[GPU CONFIG] 💡 Maximum CPU memory - for heavy FLUX processing or large system RAM")
            
        print(f"[GPU CONFIG] Output config: {config}") 