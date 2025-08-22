from typing import List, Dict, Any
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterList


class LoraSet(DataNode):
    """Collect multiple LoRA objects into a reusable set for FLUX inference"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Flux MLX"
        self.description = (
            "Collect multiple LoRA objects into a set for reuse across inference nodes"
        )

        # Input parameter list for collecting LoRAs
        self.add_parameter(
            ParameterList(
                name="loras",
                input_types=["dict", "list[dict]"],
                default_value=[],
                tooltip="Connect multiple LoRA objects from HuggingFace LoRA Discovery or LoRA Loader nodes.\nEach LoRA dict should contain 'path' and 'scale' keys.",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "LoRA Models", "hide_property": True},
            )
        )

        # Optional set name for organization
        self.add_parameter(
            Parameter(
                name="set_name",
                type="str",
                default_value="",
                tooltip="Optional name for this LoRA set (for organization)",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Set Name",
                    "placeholder_text": "e.g., Portrait Style LoRAs",
                },
            )
        )

        # Global scale multiplier
        self.add_parameter(
            Parameter(
                name="global_scale_multiplier",
                type="float",
                default_value=1.0,
                tooltip="Multiply all LoRA scales by this factor (1.0 = no change)",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"display_name": "Global Scale Multiplier"},
            )
        )

        # Output parameters
        self.add_parameter(
            Parameter(
                name="lora_set",
                output_type="list[dict]",
                type="list[dict]",
                tooltip="Collection of LoRA configurations for inference nodes",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="status",
                output_type="str",
                type="str",
                tooltip="Set composition and statistics",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={
                    "display_name": "Status",
                    "multiline": True,
                    "is_full_width": True,
                },
            )
        )

    def validate_before_node_run(
        self, exceptions: list[Exception] = None
    ) -> list[Exception] | None:
        """Validate the LoRA inputs"""
        if exceptions is None:
            exceptions = []

        try:
            loras = self.get_parameter_list_value("loras")

            if not loras:
                exceptions.append(
                    Exception("Please connect at least one LoRA to create a set")
                )
                return exceptions

            # Validate each LoRA dict has required fields
            for i, lora in enumerate(loras):
                if not isinstance(lora, dict):
                    exceptions.append(
                        Exception(f"LoRA {i + 1} is not a valid dict object")
                    )
                    continue

                if "path" not in lora:
                    exceptions.append(Exception(f"LoRA {i + 1} missing 'path' field"))

                if "scale" not in lora:
                    exceptions.append(Exception(f"LoRA {i + 1} missing 'scale' field"))
                elif not isinstance(lora["scale"], (int, float)):
                    exceptions.append(
                        Exception(f"LoRA {i + 1} 'scale' must be a number")
                    )

        except Exception as e:
            exceptions.append(Exception(f"Error validating LoRA inputs: {str(e)}"))

        return exceptions if exceptions else None

    def process(self) -> None:
        """Process the LoRA collection and create output set"""
        try:
            loras = self.get_parameter_list_value("loras")
            set_name = self.get_parameter_value("set_name")
            global_multiplier = float(
                self.get_parameter_value("global_scale_multiplier")
            )

            if not loras:
                # Handle empty case
                self.parameter_output_values["lora_set"] = []
                self.parameter_output_values["status"] = "⚠️ No LoRAs connected to set"
                return

            # Process and validate LoRAs
            processed_loras = []
            stats = {
                "total_loras": len(loras),
                "local_loras": 0,
                "hf_loras": 0,
                "total_scale": 0.0,
                "avg_scale": 0.0,
                "types": set(),
            }

            for i, lora in enumerate(loras):
                if (
                    not isinstance(lora, dict)
                    or "path" not in lora
                    or "scale" not in lora
                ):
                    print(f"[LORA SET] Warning: Skipping invalid LoRA {i + 1}: {lora}")
                    continue

                # Create a copy and apply global multiplier
                processed_lora = lora.copy()
                original_scale = float(processed_lora["scale"])
                new_scale = original_scale * global_multiplier
                processed_lora["scale"] = new_scale

                processed_loras.append(processed_lora)

                # Update statistics
                lora_type = processed_lora.get("type", "unknown")
                stats["types"].add(lora_type)
                stats["total_scale"] += new_scale

                if lora_type == "local":
                    stats["local_loras"] += 1
                elif lora_type == "huggingface":
                    stats["hf_loras"] += 1

                print(
                    f"[LORA SET] Added LoRA {i + 1}: {processed_lora.get('name', 'Unknown')} (scale: {original_scale} → {new_scale})"
                )

            # Calculate average scale
            if processed_loras:
                stats["avg_scale"] = stats["total_scale"] / len(processed_loras)

            # Set outputs
            self.parameter_output_values["lora_set"] = processed_loras

            # Create status message
            status_lines = []

            if set_name.strip():
                status_lines.append(f"📦 LoRA Set: {set_name.strip()}")
            else:
                status_lines.append(f"📦 LoRA Set (unnamed)")

            status_lines.extend(
                [
                    f"📊 Total LoRAs: {len(processed_loras)}",
                    f"🏠 Local Files: {stats['local_loras']}",
                    f"🤗 HuggingFace: {stats['hf_loras']}",
                    f"⚖️ Average Scale: {stats['avg_scale']:.2f}",
                ]
            )

            if global_multiplier != 1.0:
                status_lines.append(f"🔧 Global Multiplier: {global_multiplier}x")

            status_lines.append("")
            status_lines.append("📋 LoRA Details:")

            for i, lora in enumerate(processed_loras):
                name = lora.get("name", "Unknown")
                scale = lora.get("scale", 0.0)
                lora_type = lora.get("type", "unknown")
                type_icon = (
                    "🏠"
                    if lora_type == "local"
                    else "🤗"
                    if lora_type == "huggingface"
                    else "❓"
                )
                status_lines.append(
                    f"  {i + 1}. {type_icon} {name} (scale: {scale:.2f})"
                )

            status_lines.extend(
                [
                    "",
                    "💡 Connect to FLUX node's LoRA parameter list",
                    "🔄 Reuse this set across multiple inference nodes",
                ]
            )

            self.parameter_output_values["status"] = "\n".join(status_lines)

            print(f"[LORA SET] Created set with {len(processed_loras)} LoRAs")
            if set_name.strip():
                print(f"[LORA SET] Set name: {set_name.strip()}")

        except Exception as e:
            # Set safe defaults
            self.parameter_output_values["lora_set"] = []
            self.parameter_output_values["status"] = (
                f"❌ Error creating LoRA set: {str(e)}"
            )
            raise Exception(f"Error in LoRA set: {str(e)}")
