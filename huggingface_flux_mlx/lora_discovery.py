from typing import Dict, List, Any
from pathlib import Path
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

class HuggingFaceLoRADiscovery(DataNode):
    """Discover FLUX LoRA models from HuggingFace cache"""
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Flux MLX"
        self.description = "Discover and select FLUX LoRA models from your HuggingFace cache with metadata"
        
        # Scan for available LoRAs and create choices
        available_loras = self._scan_available_loras()
        
        self.add_parameter(
            Parameter(
                name="selected_lora",
                tooltip="Select a LoRA model from your HuggingFace cache",
                type="str",
                default_value=available_loras[0] if available_loras else "",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=available_loras)},
                ui_options={"display_name": "LoRA Model"}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="lora_scale",
                tooltip="Strength of the LoRA effect (0.0 = no effect, 1.0 = full effect)",
                type="float",
                default_value=1.0,
                allowed_modes={ParameterMode.PROPERTY},
                traits={Slider(min_val=0.0, max_val=1.0)},
                ui_options={"display_name": "LoRA Strength"}
            )
        )
        
        # Output the LoRA as a structured dict
        self.add_parameter(
            Parameter(
                name="lora_dict",
                output_type="dict",
                tooltip="LoRA information as dict with path, scale, and metadata",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "LoRA Dict"}
            )
        )
        
        # Also output just the path for simple connections
        self.add_parameter(
            Parameter(
                name="lora_path", 
                output_type="str",
                tooltip="HuggingFace repository path for the selected LoRA",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "LoRA Path"}
            )
        )
        
        # Status output for user feedback
        self.add_parameter(
            Parameter(
                name="status",
                output_type="str",
                tooltip="Discovery status and LoRA information",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Status", "multiline": True}
            )
        )
    
    def _scan_available_loras(self) -> List[str]:
        """Scan HuggingFace cache for FLUX LoRA repositories"""
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            print("[LoRA SCAN] HuggingFace Hub not available")
            return ["No HuggingFace models found"]
        
        available_loras = []
        
        try:
            cache_info = scan_cache_dir()
            
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                
                # Check if this looks like a LoRA repository
                if self._is_lora_repository(repo_id):
                    # Verify it has actual model files
                    if len(repo.revisions) > 0:
                        latest_revision = next(iter(repo.revisions))
                        snapshot_path = Path(latest_revision.snapshot_path)
                        
                        # Check for LoRA files (.safetensors)
                        safetensors_files = list(snapshot_path.glob("*.safetensors"))
                        if safetensors_files:
                            available_loras.append(repo_id)
                            print(f"[LoRA SCAN] Found LoRA: {repo_id}")
                        else:
                            print(f"[LoRA SCAN] Skipping {repo_id}: No .safetensors files")
                    
        except Exception as e:
            print(f"[LoRA SCAN] Error scanning cache: {e}")
            
        # Fallback if no LoRAs found
        if not available_loras:
            available_loras = ["No LoRA models found in cache"]
            print("[LoRA SCAN] No LoRA models found. Download some LoRAs with: huggingface-cli download")
        
        print(f"[LoRA SCAN] Available LoRAs: {available_loras}")
        return available_loras
    
    def _is_lora_repository(self, repo_id: str) -> bool:
        """Check if repository appears to be a LoRA model"""
        repo_lower = repo_id.lower()
        
        # General LoRA patterns
        general_lora_patterns = [
            "lora" in repo_lower,
            "-lora-" in repo_lower,
            "lora_" in repo_lower,
            repo_lower.endswith("_lora"),
            repo_lower.endswith("-lora"),
            "adapter" in repo_lower
        ]
        
        # FLUX-specific LoRA patterns
        flux_lora_patterns = [
            ("flux" in repo_lower and "lora" in repo_lower),
            ("flux.1" in repo_lower and "lora" in repo_lower),
            ("flux-lora" in repo_lower),
            ("flux_lora" in repo_lower)
        ]
        
        # Exclude base models and other types
        exclusion_patterns = [
            "base" in repo_lower and "model" in repo_lower,
            "transformer" in repo_lower and not "lora" in repo_lower,
            "text_encoder" in repo_lower,
            "vae" in repo_lower and not "lora" in repo_lower,
            "controlnet" in repo_lower and not "lora" in repo_lower
        ]
        
        # Must match LoRA patterns and not match exclusions
        has_lora_pattern = any(general_lora_patterns + flux_lora_patterns)
        has_exclusion = any(exclusion_patterns)
        
        return has_lora_pattern and not has_exclusion
    
    def _get_lora_metadata(self, repo_id: str) -> Dict[str, Any]:
        """Extract metadata about the LoRA model"""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Get model info
            model_info = api.model_info(repo_id)
            
            # Extract display name (prefer model card title, fallback to repo name)
            display_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
            if hasattr(model_info, 'card_data') and model_info.card_data:
                if hasattr(model_info.card_data, 'title') and model_info.card_data.title:
                    display_name = model_info.card_data.title
            
            # Extract description
            description = "FLUX LoRA model"
            if hasattr(model_info, 'description') and model_info.description:
                description = model_info.description[:100] + "..." if len(model_info.description) > 100 else model_info.description
            
            # Check if model is gated
            is_gated = getattr(model_info, 'gated', False)
            
            # Get download count if available
            downloads = getattr(model_info, 'downloads', 0)
            
            return {
                "display_name": display_name,
                "description": description,
                "gated": is_gated,
                "downloads": downloads,
                "tags": getattr(model_info, 'tags', [])
            }
            
        except Exception as e:
            print(f"[LoRA SCAN] Error getting metadata for {repo_id}: {e}")
            return {
                "display_name": repo_id.split("/")[-1] if "/" in repo_id else repo_id,
                "description": "FLUX LoRA model",
                "gated": False,
                "downloads": 0,
                "tags": []
            }
    
    def process(self) -> None:
        """Process the selected LoRA and create output dict"""
        try:
            selected_lora = self.get_parameter_value("selected_lora")
            lora_scale = float(self.get_parameter_value("lora_scale"))
            
            # Handle case where no LoRAs were found
            if selected_lora == "No LoRA models found in cache":
                self.parameter_output_values["lora_dict"] = {}
                self.parameter_output_values["lora_path"] = ""
                self.parameter_output_values["status"] = "❌ No LoRA models found in HuggingFace cache.\n💡 Download LoRAs with: huggingface-cli download <repo-id>"
                return
            
            # Get metadata for the selected LoRA
            metadata = self._get_lora_metadata(selected_lora)
            
            # Create the LoRA dict
            lora_dict = {
                "path": selected_lora,
                "scale": lora_scale,
                "name": metadata["display_name"],
                "description": metadata["description"],
                "type": "huggingface",
                "gated": metadata["gated"],
                "downloads": metadata["downloads"],
                "tags": metadata["tags"]
            }
            
            # Set outputs
            self.parameter_output_values["lora_dict"] = lora_dict
            self.parameter_output_values["lora_path"] = selected_lora
            
            # Create status message
            status_lines = [
                f"✅ LoRA Selected: {metadata['display_name']}",
                f"📁 Repository: {selected_lora}",
                f"⚖️ Strength: {lora_scale}",
                f"📝 Description: {metadata['description']}"
            ]
            
            if metadata["gated"]:
                status_lines.append("🔒 Note: This model is gated and may require approval")
            
            if metadata["downloads"] > 0:
                status_lines.append(f"📈 Downloads: {metadata['downloads']:,}")
            
            if metadata["tags"]:
                relevant_tags = [tag for tag in metadata["tags"][:5] if not tag.startswith("diffusers")]
                if relevant_tags:
                    status_lines.append(f"🏷️ Tags: {', '.join(relevant_tags)}")
            
            status_lines.append(f"\n💡 Connect to FLUX node's LoRA parameter list")
            
            self.parameter_output_values["status"] = "\n".join(status_lines)
            
        except Exception as e:
            # Set safe defaults
            self.parameter_output_values["lora_dict"] = {}
            self.parameter_output_values["lora_path"] = ""
            self.parameter_output_values["status"] = f"❌ Error processing LoRA: {str(e)}"
            raise Exception(f"Error in LoRA discovery: {str(e)}") 