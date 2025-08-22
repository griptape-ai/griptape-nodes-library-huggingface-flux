import json
import os
from typing import Any
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult

# Optional dependency - import within methods to avoid import errors
SERVICE = "Huggingface"
API_KEY_ENV_VAR = "HUGGINGFACE_HUB_ACCESS_TOKEN"


class HuggingFaceModelDownload(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "HuggingFace"
        self.description = "Download HuggingFace models to local cache with real-time status updates. Downloads model files, config, and model card."

        # Input parameter for model ID
        self.add_parameter(
            Parameter(
                name="repo_id",
                input_types=["str"],
                type="str",
                tooltip="HuggingFace model repository ID (e.g. 'nateraw/vit-base-beans')",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"placeholder_text": "e.g., microsoft/DialoGPT-medium"},
            )
        )

        # Output parameters
        self.add_parameter(
            Parameter(
                name="download_status",
                output_type="str",
                type="str",
                tooltip="Real-time download status and progress",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Status", "multiline": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="download_path",
                output_type="str",
                type="str",
                tooltip="Local path where model was downloaded",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Download Path"},
            )
        )

        self.add_parameter(
            Parameter(
                name="model_card_text",
                output_type="str",
                type="str",
                tooltip="Model card README content",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Model Card", "multiline": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="model_card_metadata",
                output_type="str",
                type="str",
                tooltip="Model card metadata as JSON",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Model Metadata", "multiline": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="files_downloaded",
                output_type="str",
                type="str",
                tooltip="List of downloaded files",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Downloaded Files", "multiline": True},
            )
        )

    def validate_node(self) -> list[Exception] | None:
        errors = []

        repo_id = self.get_parameter_value("repo_id")
        if not repo_id or not repo_id.strip():
            errors.append(
                ValueError(
                    "Repository ID is required (e.g., 'microsoft/DialoGPT-medium')"
                )
            )

        return errors if errors else None

    def process(self) -> AsyncResult:
        # Validate before starting
        validation_errors = self.validate_node()
        if validation_errors:
            error_message = "; ".join(str(e) for e in validation_errors)
            raise ValueError(f"Validation failed: {error_message}")

        def download_model() -> str:
            try:
                # Import dependencies
                from huggingface_hub import (
                    snapshot_download,
                    ModelCard,
                    hf_hub_download,
                )

                try:
                    from tqdm import tqdm

                    tqdm_available = True
                except ImportError:
                    tqdm_available = False
            except ImportError:
                error_msg = "huggingface_hub library not installed. Please add 'huggingface_hub' to your library dependencies."
                self.publish_update_to_parameter(
                    "download_status", f"ERROR: {error_msg}"
                )
                raise ImportError(error_msg)

            # Create a simple tqdm class for clean overall progress
            node_self = self  # Capture the node instance in closure

            if tqdm_available:

                class SimpleProgressTqdm(tqdm):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.last_update_time = 0
                        self.update_interval = 1.0  # Update every 1 second

                    def format_bytes(self, bytes_val):
                        """Format bytes into human readable format"""
                        if bytes_val == 0:
                            return "0 B"
                        for unit in ["B", "KB", "MB", "GB", "TB"]:
                            if bytes_val < 1024.0:
                                return f"{bytes_val:.1f} {unit}"
                            bytes_val /= 1024.0
                        return f"{bytes_val:.1f} PB"

                    def update(self, n=1):
                        super().update(n)

                        # Throttle updates to avoid spam
                        import time

                        current_time = time.time()
                        if current_time - self.last_update_time >= self.update_interval:
                            if hasattr(self, "total") and self.total and self.total > 0:
                                # Calculate percentage
                                percentage = min(100, (self.n / self.total) * 100)

                                # Get description
                                desc = getattr(self, "desc", "") or ""

                                # Format progress message
                                if "files" in desc.lower():
                                    # File count progress
                                    progress_msg = f"📦 {desc}: {self.n}/{self.total} files ({percentage:.1f}%)"
                                else:
                                    # Byte progress
                                    current_bytes = self.format_bytes(self.n)
                                    total_bytes = self.format_bytes(self.total)
                                    progress_msg = f"📥 {desc or 'Downloading'}: {current_bytes}/{total_bytes} ({percentage:.1f}%)"

                                    # Add speed if available
                                    try:
                                        format_dict = getattr(self, "format_dict", {})
                                        rate = format_dict.get("rate", 0)
                                        if rate and rate > 0:
                                            speed_str = self.format_bytes(rate)
                                            progress_msg += f" @ {speed_str}/s"
                                    except:
                                        pass

                                # Publish the update
                                node_self.publish_update_to_parameter(
                                    "download_status", progress_msg
                                )

                            self.last_update_time = current_time

            repo_id = self.get_parameter_value("repo_id").strip()

            # Get HF token (optional for public models)
            token = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)

            try:
                # Phase 1: Load model card (fast)
                self.publish_update_to_parameter(
                    "download_status", "Loading model card..."
                )

                try:
                    card = ModelCard.load(repo_id, token=token if token else None)
                    model_card_text = card.text or "No model card text available"

                    # Get metadata as JSON
                    try:
                        metadata_dict = card.data.to_dict() if card.data else {}
                        model_card_metadata = json.dumps(metadata_dict, indent=2)
                    except Exception:
                        model_card_metadata = "{}"

                except Exception as e:
                    model_card_text = f"Could not load model card: {str(e)}"
                    model_card_metadata = "{}"

                # Update model card outputs
                self.publish_update_to_parameter("model_card_text", model_card_text)
                self.publish_update_to_parameter(
                    "model_card_metadata", model_card_metadata
                )

                # Phase 2: Download repository (slow)
                self.publish_update_to_parameter(
                    "download_status",
                    "Starting model download...\nThis may take several minutes for large models.",
                )

                # Essential file patterns for models
                allow_patterns = [
                    "*.json",  # Config files
                    "*.safetensors",  # Model weights (preferred format)
                    "*.bin",  # PyTorch model weights
                    "*.txt",  # Text files (vocab, etc.)
                    "*.py",  # Modeling code
                    "README.md",  # Model card
                    "*.md",  # Other documentation
                    "tokenizer*",  # Tokenizer files
                    "*.model",  # SentencePiece/other model files
                ]

                # Download to default HF cache with progress
                if tqdm_available:
                    download_path = snapshot_download(
                        repo_id=repo_id,
                        token=token if token else None,
                        allow_patterns=allow_patterns,
                        tqdm_class=SimpleProgressTqdm,
                    )
                else:
                    # Fallback without progress if tqdm not available
                    self.publish_update_to_parameter(
                        "download_status",
                        "📥 Downloading (progress tracking unavailable - install tqdm)...",
                    )
                    download_path = snapshot_download(
                        repo_id=repo_id,
                        token=token if token else None,
                        allow_patterns=allow_patterns,
                    )

                # Get list of downloaded files
                downloaded_files = []
                if os.path.exists(download_path):
                    for root, dirs, files in os.walk(download_path):
                        for file in files:
                            rel_path = os.path.relpath(
                                os.path.join(root, file), download_path
                            )
                            downloaded_files.append(rel_path)

                files_list = (
                    "\n".join(sorted(downloaded_files))
                    if downloaded_files
                    else "No files found"
                )

                # Update outputs
                self.publish_update_to_parameter("download_path", download_path)
                self.publish_update_to_parameter("files_downloaded", files_list)

                # Final status
                final_status = f"✅ Download complete!\n\nModel: {repo_id}\nPath: {download_path}\nFiles: {len(downloaded_files)} files"
                self.publish_update_to_parameter("download_status", final_status)

                return final_status

            except Exception as e:
                error_msg = f"❌ Download failed: {str(e)}"
                self.publish_update_to_parameter("download_status", error_msg)
                self.publish_update_to_parameter("download_path", "")
                self.publish_update_to_parameter("files_downloaded", "")
                raise Exception(error_msg)

        # Return the generator for async processing
        yield download_model
