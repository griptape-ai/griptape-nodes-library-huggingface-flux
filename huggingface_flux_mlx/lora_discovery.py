from typing import Dict, List, Any
from pathlib import Path
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

# Note: LoRA discovery uses local config patterns to avoid import complexity


class HuggingFaceLoRADiscovery(DataNode):
    """Discover FLUX LoRA models from HuggingFace cache"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Flux MLX"
        self.description = "Discover and select FLUX LoRA models from your HuggingFace cache with metadata"

        # Configuration patterns (avoid complex imports in discovery node)
        self.repository_patterns = {
            "lora_keywords": ["flux.1", "lora"],
            "encoder_repos": ["comfyanonymous"],
        }

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
                ui_options={"display_name": "LoRA Model"},
            )
        )

        self.add_parameter(
            Parameter(
                name="lora_scale",
                tooltip="Strength of the LoRA effect (0.0 = no effect, 1.0 = full effect)",
                type="float",
                default_value=1.0,
                allowed_modes={ParameterMode.PROPERTY},
                traits={Slider(min_val=0.0, max_val=2.0)},
                ui_options={"display_name": "LoRA Strength"},
            )
        )

        # Trigger words extracted from model card - moved up under strength
        self.add_parameter(
            Parameter(
                name="trigger_words",
                output_type="str",
                tooltip="Trigger words found in the LoRA's model card",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Trigger Words"},
            )
        )

        # Output the LoRA as a structured dict - keep output but hide UI display
        self.add_parameter(
            Parameter(
                name="lora_dict",
                output_type="dict",
                tooltip="LoRA information as dict with path, scale, and metadata",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "LoRA Config", "hide_property": True},
            )
        )

        # Also output just the path for simple connections
        self.add_parameter(
            Parameter(
                name="lora_path",
                output_type="str",
                tooltip="HuggingFace repository path for the selected LoRA",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "LoRA Path"},
            )
        )

        # Status output for user feedback
        self.add_parameter(
            Parameter(
                name="status",
                output_type="str",
                tooltip="Discovery status and LoRA information",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Status", "multiline": True},
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
                            print(
                                f"[LoRA SCAN] Skipping {repo_id}: No .safetensors files"
                            )

        except Exception as e:
            print(f"[LoRA SCAN] Error scanning cache: {e}")

        # Fallback if no LoRAs found
        if not available_loras:
            available_loras = ["No LoRA models found in cache"]
            print(
                "[LoRA SCAN] No LoRA models found. Download some LoRAs with: huggingface-cli download"
            )

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
            "adapter" in repo_lower,
        ]

        # FLUX-specific LoRA patterns (use local config)
        lora_keywords = self.repository_patterns.get(
            "lora_keywords", ["flux.1", "lora"]
        )

        flux_lora_patterns = [
            all(
                keyword in repo_lower for keyword in lora_keywords[:2]
            ),  # Both flux.1 and lora
            ("flux" in repo_lower and "lora" in repo_lower),
            ("flux-lora" in repo_lower),
            ("flux_lora" in repo_lower),
        ]

        # Exclude base models and other types
        exclusion_patterns = [
            "base" in repo_lower and "model" in repo_lower,
            "transformer" in repo_lower and not "lora" in repo_lower,
            "text_encoder" in repo_lower,
            "vae" in repo_lower and not "lora" in repo_lower,
            "controlnet" in repo_lower and not "lora" in repo_lower,
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
            if hasattr(model_info, "card_data") and model_info.card_data:
                if (
                    hasattr(model_info.card_data, "title")
                    and model_info.card_data.title
                ):
                    display_name = model_info.card_data.title

            # Extract description
            description = "FLUX LoRA model"
            if hasattr(model_info, "description") and model_info.description:
                description = (
                    model_info.description[:100] + "..."
                    if len(model_info.description) > 100
                    else model_info.description
                )

            # Check if model is gated
            is_gated = getattr(model_info, "gated", False)

            # Get download count if available
            downloads = getattr(model_info, "downloads", 0)

            return {
                "display_name": display_name,
                "description": description,
                "gated": is_gated,
                "downloads": downloads,
                "tags": getattr(model_info, "tags", []),
            }

        except Exception as e:
            print(f"[LoRA SCAN] Error getting metadata for {repo_id}: {e}")
            return {
                "display_name": repo_id.split("/")[-1] if "/" in repo_id else repo_id,
                "description": "FLUX LoRA model",
                "gated": False,
                "downloads": 0,
                "tags": [],
            }

    def _is_valid_trigger(self, text: str) -> bool:
        """Check if extracted text is a valid trigger word"""
        import re

        if not text or len(text) < 3:
            return False

        # Filter out HTML/CSS/technical fragments
        invalid_patterns = [
            r"<[^>]+>",  # HTML tags
            r"style\s*=",  # CSS style attributes
            r"div\s*>",  # HTML div fragments
            r"img\s*>",  # HTML img fragments
            r"\.(safetensors|ckpt|pt)$",  # File extensions
            r"^[{}()[\];,.<>]+$",  # Only punctuation
            r"^\s*\)\s*$",  # Just parentheses
            r"^\s*=\s*$",  # Just equals
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False

        # Must be reasonable length
        if len(text) > 100:
            return False

        # Should contain actual words
        if not re.search(r"[a-zA-Z]", text):
            return False

        return True

    def _clean_trigger_phrase(self, raw_trigger: str) -> str:
        """Extract clean trigger phrase from raw matched text"""
        import re

        # Remove common instructional phrases and extract core trigger
        cleaned = raw_trigger.strip()

        # Handle specific instructional patterns first
        instructional_patterns = [
            r"^.*?you\s+should\s+use\s+(.+?)\s+to\s+trigger.*$",  # "You should use X to trigger"
            r"^.*?use\s+(.+?)\s+to\s+trigger.*$",  # "use X to trigger"
            r"^.*?use\s+(.+?)\s+for.*$",  # "use X for"
            r"^.*?trigger\s+words?:?\s*(.+?)$",  # "trigger words: X"
        ]

        for pattern in instructional_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
                break

        # Remove common ending phrases
        endings_to_remove = [
            r"\s+to\s+trigger.*$",
            r"\s+the\s+image\s+generation.*$",
            r"\s+generation.*$",
            r"\s+for\s+.*$",
            r"\s+when\s+.*$",
            r"\s+in\s+.*$",
        ]

        for ending in endings_to_remove:
            cleaned = re.sub(ending, "", cleaned, flags=re.IGNORECASE)

        # Clean up remaining prefixes
        prefixes_to_remove = [
            r"^you\s+should\s+use\s+",
            r"^use\s+",
            r"^trigger\s+words?:?\s*",
            r"^activation\s+words?:?\s*",
        ]

        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

        # Clean up extra whitespace and punctuation
        cleaned = re.sub(r'[*_`"\']', "", cleaned.strip())
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()

        return cleaned

    def _extract_trigger_words(self, repo_id: str) -> str:
        """Extract trigger words from LoRA model card"""
        import re

        try:
            from huggingface_hub import HfApi

            api = HfApi()

            # Try to get README.md content
            readme_content = api.hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model",
                local_files_only=False,
            )

            with open(readme_content, "r", encoding="utf-8") as f:
                model_card = f.read()

            trigger_words = []

            # Pattern 1: Explicit "trigger words" sections
            trigger_patterns = [
                r"##\s*trigger\s*words?\s*\n+(.*?)(?=\n##|\Z)",
                r"trigger\s*words?:\s*(.+?)(?=\n|$)",
                r"trigger\s*phrases?:\s*(.+?)(?=\n|$)",
                r"activation\s*words?:\s*(.+?)(?=\n|$)",
                r"you\s+should\s+use\s+(.+?)\s+to\s+trigger",
            ]

            for pattern in trigger_patterns:
                matches = re.findall(pattern, model_card, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    # Clean and extract core trigger phrase
                    cleaned = self._clean_trigger_phrase(match)
                    if self._is_valid_trigger(cleaned):
                        trigger_words.append(cleaned)

            # Pattern 2: Code blocks with style keywords
            code_block_patterns = [
                r"`([^`]+style[^`]*)`",
                r'"([^"]+style[^"]*)"',
                r"'([^']+style[^']*)'",
            ]

            for pattern in code_block_patterns:
                matches = re.findall(pattern, model_card, re.IGNORECASE)
                for match in matches:
                    cleaned = self._clean_trigger_phrase(match)
                    if self._is_valid_trigger(cleaned) and cleaned not in trigger_words:
                        trigger_words.append(cleaned)

            # Return the first valid trigger word found, or empty string
            return trigger_words[0] if trigger_words else ""

        except Exception as e:
            print(f"[TRIGGER] Could not extract triggers for {repo_id}: {e}")
            return ""

    def process(self) -> None:
        """Process the selected LoRA and create output dict"""
        try:
            selected_lora = self.get_parameter_value("selected_lora")
            lora_scale = float(self.get_parameter_value("lora_scale"))

            # Handle case where no LoRAs were found
            if selected_lora == "No LoRA models found in cache":
                self.parameter_output_values[
                    "lora_dict"
                ] = {}  # Empty dict for no LoRAs
                self.parameter_output_values["lora_path"] = ""
                self.parameter_output_values["trigger_words"] = ""
                self.parameter_output_values["status"] = (
                    "❌ No LoRA models found in HuggingFace cache.\n💡 Download LoRAs with: huggingface-cli download <repo-id>"
                )
                return

            # Get metadata for the selected LoRA
            metadata = self._get_lora_metadata(selected_lora)

            # Extract trigger words from model card
            trigger_words = self._extract_trigger_words(selected_lora)

            # Create the LoRA dict
            lora_dict = {
                "path": selected_lora,
                "scale": lora_scale,
                "name": metadata["display_name"],
                "description": metadata["description"],
                "type": "huggingface",
                "gated": metadata["gated"],
                "downloads": metadata["downloads"],
                "tags": metadata["tags"],
            }

            # Set outputs - output dict directly for ParameterList
            self.parameter_output_values["lora_dict"] = lora_dict

            # Simple debug output
            print(
                f"[LORA DEBUG] Selected: {metadata['display_name']} (scale: {lora_scale})"
            )
            self.parameter_output_values["lora_path"] = selected_lora
            self.parameter_output_values["trigger_words"] = trigger_words

            # Create status message
            status_lines = [
                f"✅ LoRA Selected: {metadata['display_name']}",
                f"📁 Repository: {selected_lora}",
                f"⚖️ Strength: {lora_scale}",
                f"📝 Description: {metadata['description']}",
            ]

            if metadata["gated"]:
                status_lines.append(
                    "🔒 Note: This model is gated and may require approval"
                )

            if metadata["downloads"] > 0:
                status_lines.append(f"📈 Downloads: {metadata['downloads']:,}")

            if metadata["tags"]:
                relevant_tags = [
                    tag
                    for tag in metadata["tags"][:5]
                    if not tag.startswith("diffusers")
                ]
                if relevant_tags:
                    status_lines.append(f"🏷️ Tags: {', '.join(relevant_tags)}")

            if trigger_words:
                status_lines.append(f"🎯 Trigger Words: {trigger_words}")

            status_lines.append(f"\n💡 Connect to FLUX node's LoRA parameter list")

            self.parameter_output_values["status"] = "\n".join(status_lines)

        except Exception as e:
            # Set safe defaults (empty dict for ParameterList compatibility)
            self.parameter_output_values["lora_dict"] = {}
            self.parameter_output_values["lora_path"] = ""
            self.parameter_output_values["trigger_words"] = ""
            self.parameter_output_values["status"] = (
                f"❌ Error processing LoRA: {str(e)}"
            )
            raise Exception(f"Error in LoRA discovery: {str(e)}")
