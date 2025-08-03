"""
Parameter builder for FLUX inference node.

Handles creation of UI parameter groups and individual parameters.
"""

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup, ParameterList
from griptape_nodes.traits.options import Options

logger = logging.getLogger(__name__)


class FluxParameterBuilder:
    """Builds UI parameters for FLUX inference node."""
    
    def __init__(self, node):
        """Initialize with the parent node."""
        self.node = node
    
    def build_all_parameters(self, backend_name: str) -> None:
        """Build all parameter groups for the FLUX inference node."""
        self._build_model_selection_group(backend_name)
        self._build_generation_settings_group()
        self._build_output_parameters()
    
    def _build_model_selection_group(self, backend_name: str) -> None:
        """Build the Model Selection parameter group."""
        # Model Selection Group - Always visible
        with ParameterGroup(name=f"Model Selection ({backend_name})") as model_group:
            
            # Main prompt (single string)
            self.node.add_parameter(
                Parameter(
                    name="main_prompt",
                    tooltip="Primary text description for image generation",
                    type="str",
                    input_types=["str"],
                    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                    ui_options={
                        "display_name": "Main Prompt",
                        "multiline": True,
                        "placeholder_text": "Describe the image you want to generate..."
                    }
                )
            )
            
            # Additional prompts (list input)
            self.node.add_parameter(
                ParameterList(
                    name="additional_prompts",
                    input_types=["str", "list[str]"],
                    default_value=[],
                    tooltip="Optional additional prompts to merge with the main prompt.",
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": "Additional Prompts"}
                )
            )
            
            self.node.add_parameter(
                Parameter(
                    name="system_constraints",
                    tooltip="Optional system constraints from GPU Configuration node. Supports: 'gpu_memory_gb' (general limit), '4-bit_memory_gb'/'8-bit_memory_gb'/'none_memory_gb' (quantization-specific limits), 'allow_low_memory' (bypass safety checks), 'gpu_device' (device selection).",
                    type="dict",
                    input_types=["dict"],
                    default_value={},
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": "GPU Configuration",  "hide_property": True}
                )
            )
            
            self.node.add_parameter(
                Parameter(
                    name="flux_config",
                    tooltip="Flux configuration dict from selector",
                    type="dict",
                    input_types=["dict"],
                    default_value={},
                    allowed_modes={ParameterMode.INPUT},
                    ui_options={"display_name": "Flux Model", "hide_property": True}
                )
            )
        
        model_group.ui_options = {"collapsed": False}
        self.node.add_node_element(model_group)
    
    def _build_generation_settings_group(self) -> None:
        """Build the Generation Settings parameter group."""
        # Shared Generation Settings - Always visible  
        with ParameterGroup(name="Generation Settings") as gen_group:
            self.node.add_parameter(
                Parameter(
                    name="width",
                    tooltip="Width of generated image in pixels",
                    type="int",
                    default_value=1024,
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=[512, 768, 1024, 1152, 1280])},
                    ui_options={"display_name": "Width"}
                )
            )
            
            self.node.add_parameter(
                Parameter(
                    name="height", 
                    tooltip="Height of generated image in pixels",
                    type="int",
                    default_value=1024,
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=[512, 768, 1024, 1152, 1280])},
                    ui_options={"display_name": "Height"}
                )
            )
            
            self.node.add_parameter(
                Parameter(
                    name="guidance_scale",
                    tooltip="How closely to follow the prompt (higher = more adherence)",
                    type="float",
                    default_value=7.5,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Guidance Scale"}
                )
            )
            
            self.node.add_parameter(
                Parameter(
                    name="steps",
                    tooltip="Number of inference steps. More steps = higher quality but slower generation. FLUX.1-dev: 15-50 steps, FLUX.1-schnell: 1-8 steps recommended.",
                    type="int",
                    default_value=20,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Inference Steps"}
                )
            )
            
            self.node.add_parameter(
                Parameter(
                    name="seed",
                    tooltip="Random seed for reproducible generation (-1 for random)",
                    type="int", 
                    default_value=-1,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Seed"}
                )
            )
            
            # Scheduler / sampler parameters
            self.node.add_parameter(
                Parameter(
                    name="scheduler",
                    tooltip="Sampling scheduler (FLUX-compatible only)",
                    type="str",
                    default_value="FlowMatchEuler",
                    allowed_modes={ParameterMode.PROPERTY},
                    traits={Options(choices=[
                        "FlowMatchEuler",
                        "Euler"
                    ])},
                    ui_options={"display_name": "Scheduler"}
                )
            )
            
            self.node.add_parameter(
                Parameter(
                    name="cfg_rescale",
                    tooltip="CFG rescale (0 disables)",
                    type="float",
                    default_value=0.0,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "CFG Rescale"}
                )
            )
            
            self.node.add_parameter(
                Parameter(
                    name="denoise_eta",
                    tooltip="DDIM / noise eta",
                    type="float",
                    default_value=0.0,
                    allowed_modes={ParameterMode.PROPERTY},
                    ui_options={"display_name": "Denoise Eta"}
                )
            )
        
        gen_group.ui_options = {"collapsed": False}
        self.node.add_node_element(gen_group)
    
    def _build_output_parameters(self) -> None:
        """Build the output parameters."""
        # Output parameters
        self.node.add_parameter(
            Parameter(
                name="image",
                output_type="ImageUrlArtifact",
                tooltip="Generated image",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Generated Image"}
            )
        )
        
        self.node.add_parameter(
            Parameter(
                name="generation_info",
                output_type="str",
                tooltip="Generation metadata and parameters used",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Generation Info", "multiline": True}
            )
        )
        
        self.node.add_parameter(
            Parameter(
                name="status",
                output_type="str", 
                tooltip="Real-time generation status and progress",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Status", "multiline": True}
            )
        )