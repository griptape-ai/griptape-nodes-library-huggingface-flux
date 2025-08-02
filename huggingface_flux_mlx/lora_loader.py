from pathlib import Path
from typing import Dict, Any
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.slider import Slider


class LoRALoader(DataNode):
    """Load local LoRA .safetensors files for FLUX inference"""
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Flux MLX"
        self.description = "Load local LoRA .safetensors files with configurable strength"
        
        # File picker for LoRA safetensors file
        lora_file = Parameter(
            name="lora_file",
            type="str",
            default_value="",
            tooltip="Select a LoRA .safetensors file from your local filesystem",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"clickable_file_browser": True, "display_name": "LoRA File"}
        )
        lora_file.add_trait(FileSystemPicker(allow_files=True, allow_directories=False))
        self.add_parameter(lora_file)
        
        # Scale parameter
        self.add_parameter(
            Parameter(
                name="lora_scale",
                tooltip="Strength of the LoRA effect (0.0 = no effect, 1.0 = full effect)",
                type="float",
                default_value=1.0,
                allowed_modes={ParameterMode.PROPERTY},
                traits={Slider(min_val=0.0, max_val=2.0)},
                ui_options={"display_name": "LoRA Strength"}
            )
        )
        
        # Optional name override
        self.add_parameter(
            Parameter(
                name="custom_name",
                type="str",
                default_value="",
                tooltip="Optional custom name for this LoRA (defaults to filename)",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"display_name": "Custom Name", "placeholder_text": "Optional custom name"}
            )
        )
        
        # Output parameters
        self.add_parameter(
            Parameter(
                name="lora_dict",
                output_type="dict",
                type="dict",
                tooltip="LoRA configuration dict for inference nodes",
                allowed_modes={ParameterMode.OUTPUT}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="lora_path",
                output_type="str",
                type="str", 
                tooltip="Absolute path to the LoRA file",
                allowed_modes={ParameterMode.OUTPUT}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="status",
                output_type="str",
                type="str",
                tooltip="Loading status and file information",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Status", "multiline": True}
            )
        )
    
    def validate_before_node_run(self, exceptions: list[Exception] = None) -> list[Exception] | None:
        """Validate the LoRA file before processing"""
        if exceptions is None:
            exceptions = []
            
        lora_file = self.get_parameter_value("lora_file")
        
        if not lora_file:
            exceptions.append(Exception("Please select a LoRA file"))
            return exceptions
            
        file_path = Path(lora_file)
        
        # Check if file exists
        if not file_path.exists():
            exceptions.append(Exception(f"File does not exist: {lora_file}"))
            return exceptions
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            exceptions.append(Exception(f"Path is not a file: {lora_file}"))
            return exceptions
            
        # Check file extension
        if not file_path.suffix.lower() == '.safetensors':
            exceptions.append(Exception(f"File must be a .safetensors file, got: {file_path.suffix}"))
            return exceptions
            
        return exceptions if exceptions else None
    
    def process(self) -> None:
        """Process the LoRA file and create output dict"""
        try:
            lora_file = self.get_parameter_value("lora_file")
            lora_scale = float(self.get_parameter_value("lora_scale"))
            custom_name = self.get_parameter_value("custom_name")
            
            file_path = Path(lora_file)
            absolute_file_path = str(file_path.resolve())
            
            # Determine display name
            if custom_name.strip():
                display_name = custom_name.strip()
            else:
                display_name = file_path.stem  # filename without extension
            
            # Get file info for status
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Create the LoRA dict (compatible with existing inference node)
            lora_dict = {
                "path": absolute_file_path,  # Direct file path
                "scale": lora_scale,
                "name": display_name,
                "description": f"Local LoRA file: {file_path.name}",
                "type": "local",
                "gated": False,
                "downloads": 0,
                "tags": ["local", "safetensors"]
            }
            
            # Set outputs
            self.parameter_output_values["lora_dict"] = lora_dict
            self.parameter_output_values["lora_path"] = absolute_file_path
            
            # Create status message
            status_lines = [
                f"✅ LoRA Loaded: {display_name}",
                f"📁 File: {file_path.name}",
                f"📂 Path: {absolute_file_path}",
                f"⚖️ Strength: {lora_scale}",
                f"📊 Size: {file_size_mb:.1f} MB",
                f"🔧 Type: Local .safetensors file",
                "",
                f"💡 Connect to FLUX node's LoRA parameter list"
            ]
            
            self.parameter_output_values["status"] = "\n".join(status_lines)
            
            print(f"[LORA LOADER] Loaded local LoRA file: {display_name} (scale: {lora_scale})")
            print(f"[LORA LOADER] File: {absolute_file_path}")
            
        except Exception as e:
            # Set safe defaults
            self.parameter_output_values["lora_dict"] = {}
            self.parameter_output_values["lora_path"] = ""
            self.parameter_output_values["status"] = f"❌ Error loading LoRA: {str(e)}"
            raise Exception(f"Error in LoRA loader: {str(e)}") 