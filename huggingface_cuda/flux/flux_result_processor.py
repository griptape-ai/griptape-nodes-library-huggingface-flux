"""
Result processing utilities for FLUX image generation.

Handles image conversion, file saving, status formatting, and artifact creation.
"""

import io
import hashlib
import time
import tempfile
import os
import logging
from typing import Any, Tuple, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class FluxResultProcessor:
    """Processes FLUX generation results into Griptape artifacts and status updates."""
    
    def __init__(self, node):
        """Initialize with the parent node for parameter updates."""
        self.node = node
    
    def process_generation_result(self, backend_result: Any, model_id: str, seed: int, 
                                width: int, height: int, guidance_scale: float, 
                                num_steps: int) -> str:
        """Process the backend generation result into final outputs."""
        
        # Parse backend result format
        generated_image, generation_info = self._parse_backend_result(
            backend_result, model_id, seed, num_steps, guidance_scale
        )
        
        print(f"[FLUX DEBUG] ===== PROCESSING GENERATION RESULT =====")
        print(f"[FLUX DEBUG] Generated image type: {type(generated_image)}")
        print(f"[FLUX DEBUG] Generation info type: {type(generation_info)}")
        print(f"[FLUX DEBUG] Generation info keys: {generation_info.keys() if isinstance(generation_info, dict) else 'NOT_DICT'}")
        
        # Ensure we have a PIL Image
        generated_image = self._ensure_pil_image(generated_image)
        
        print(f"[FLUX DEBUG] Final image type: {type(generated_image)}")
        print(f"[FLUX DEBUG] Final image size: {generated_image.size if hasattr(generated_image, 'size') else 'NO_SIZE'}")
        
        # Save the image and get URL
        static_url = self._save_image(generated_image)
        
        # Create and update status
        final_status = self._format_status(width, height, generation_info)
        self.node.publish_update_to_parameter("status", final_status)
        print(f"[FLUX DEBUG] Final status: {final_status}")
        
        # Create and set output artifacts
        self._create_output_artifacts(static_url, generation_info)
        
        print(f"[FLUX DEBUG] ===== GENERATION COMPLETE =====")
        print(f"[FLUX DEBUG] Image URL: {static_url}")
        print(f"[FLUX DEBUG] Generation info: {generation_info}")
        return static_url
    
    def _parse_backend_result(self, backend_result: Any, model_id: str, seed: int, 
                            num_steps: int, guidance_scale: float) -> Tuple[Any, Dict[str, Any]]:
        """Parse different backend result formats into image and info."""
        
        if isinstance(backend_result, tuple) and len(backend_result) == 2:
            generated_image, generation_info = backend_result
            print(f"[FLUX DEBUG] Got tuple result: image={type(generated_image)}, info={type(generation_info)}")
            return generated_image, generation_info
        else:
            print(f"[FLUX DEBUG] Backend result format: {type(backend_result)}")
            # Handle FluxPipelineOutput from diffusers
            if hasattr(backend_result, 'images') and backend_result.images:
                generated_image = backend_result.images[0]  # First image
                generation_info = {
                    "backend": "Diffusers (CUDA)",  # Default backend name
                    "model": model_id,
                    "actual_seed": seed,
                    "steps": num_steps,
                    "guidance": guidance_scale
                }
                print(f"[FLUX DEBUG] Extracted from FluxPipelineOutput: image={type(generated_image)}")
                return generated_image, generation_info
            else:
                print(f"[FLUX DEBUG] Unexpected backend result format: {type(backend_result)}")
                raise RuntimeError(f"Backend returned unexpected format: {type(backend_result)}")
    
    def _ensure_pil_image(self, generated_image: Any) -> Any:
        """Ensure the generated image is a PIL Image."""
        if not hasattr(generated_image, 'save'):
            print(f"[FLUX DEBUG] Converting result to PIL Image...")
            try:
                from PIL import Image
                if hasattr(generated_image, 'numpy'):  # torch tensor
                    import numpy as np
                    img_array = generated_image.cpu().numpy()
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype('uint8')
                    generated_image = Image.fromarray(img_array)
                else:
                    raise RuntimeError(f"Cannot convert {type(generated_image)} to PIL Image")
            except Exception as conv_error:
                print(f"[FLUX DEBUG] Conversion failed: {conv_error}")
                raise RuntimeError(f"Failed to convert generated image to PIL format: {conv_error}")
        
        return generated_image
    
    def _save_image(self, generated_image: Any) -> str:
        """Save the image and return the static URL."""
        # Convert image to bytes
        image_bytes = io.BytesIO()
        generated_image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()
        
        # Generate unique filename
        content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        timestamp = int(time.time() * 1000)
        filename = f"flux_generated_{timestamp}_{content_hash}.png"
        
        # Save using GriptapeNodes StaticFilesManager
        try:
            from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
            static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                image_bytes, filename
            )
            print(f"[FLUX DEBUG] Image saved: {static_url}")
            return static_url
        except Exception as save_error:
            print(f"[FLUX DEBUG] StaticFilesManager failed: {save_error}")
            # Fallback to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                generated_image.save(tmp_file.name, format="PNG")
                static_url = f"file://{tmp_file.name}"
                print(f"[FLUX DEBUG] Fallback temp file: {static_url}")
                return static_url
    
    def _format_status(self, width: int, height: int, generation_info: Dict[str, Any]) -> str:
        """Format the final status message."""
        final_status = f"✅ Generated {width}x{height} image"
        if generation_info and isinstance(generation_info, dict):
            if 'actual_seed' in generation_info:
                final_status += f" (seed: {generation_info['actual_seed']})"
            if 'generation_time' in generation_info:
                final_status += f" in {generation_info['generation_time']:.1f}s"
        return final_status
    
    def _create_output_artifacts(self, static_url: str, generation_info: Dict[str, Any]) -> None:
        """Create and set the output artifacts."""
        # Try to create ImageUrlArtifact for better UX
        try:
            from griptape.artifacts import ImageUrlArtifact
            image_artifact = ImageUrlArtifact(value=static_url)
            self.node.parameter_output_values["image"] = image_artifact
            print(f"[FLUX DEBUG] Created ImageUrlArtifact successfully")
        except Exception as artifact_error:
            print(f"[FLUX DEBUG] Artifact creation failed: {artifact_error}")
            # Fallback to simple string return
            self.node.parameter_output_values["image"] = static_url
            self.node.parameter_output_values["generation_info"] = str(generation_info)
            return
            
        # Return generation info as string for the parameter
        self.node.parameter_output_values["generation_info"] = str(generation_info)


def create_error_image(tmp_dir: Path, error_msg: str, exception: Exception) -> str:
    """Create a tiny PNG with the error text and return file:// URL."""
    from PIL import Image, ImageDraw
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (600, 120), color=(30, 30, 30))
    d = ImageDraw.Draw(img)
    d.text((10, 10), error_msg[:200], fill=(255, 0, 0))
    filepath = tmp_dir / f"flux_error_{int(time.time()*1000)}.png"
    img.save(filepath)
    return f"file://{filepath}"