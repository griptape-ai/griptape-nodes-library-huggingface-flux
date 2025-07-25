"""
FluxTextEncoders Node - CLIP and T5 text encoding with GPU selection.

This node handles text encoding for FLUX models using CLIP and T5 encoders,
with full GPU device and memory allocation control.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from griptape.artifacts import BaseArtifact, TextArtifact, ListArtifact
from griptape.mixins import SerializableMixin
from griptape.utils import import_optional_dependency

from .gpu_utils import select_gpu, get_available_gpus, cleanup_gpu_memory

logger = logging.getLogger(__name__)


def _get_shared_backend():
    """Get shared backend for torch and transformers"""
    try:
        from .. import get_shared_backend
        return get_shared_backend()
    except:
        logger.error("Could not access shared backend")
        return None


@dataclass
class TextEmbeddingArtifact(BaseArtifact, SerializableMixin):
    """Artifact for storing text embeddings and metadata."""
    
    prompt_embeds: Any  # torch.Tensor - use Any to avoid import issues
    pooled_prompt_embeds: Any  # torch.Tensor
    original_prompts: List[str]
    prompt_weights: Optional[List[float]] = None
    device: str = "cpu"
    dtype: str = "float32"
    
    def __post_init__(self):
        super().__post_init__()
        
    @property
    def value(self) -> Dict[str, Any]:
        """Return the embedding data as a dictionary."""
        return {
            "prompt_embeds": self.prompt_embeds,
            "pooled_prompt_embeds": self.pooled_prompt_embeds,
            "original_prompts": self.original_prompts,
            "prompt_weights": self.prompt_weights,
            "device": self.device,
            "dtype": self.dtype
        }


class FluxTextEncoders:
    """
    FLUX Text Encoders Node with GPU selection and memory management.
    
    Handles CLIP and T5 text encoding for FLUX models with granular control
    over GPU device selection and memory allocation.
    """
    
    def __init__(self, 
                 gpu_device: int = 0,
                 memory_fraction: float = 0.4,
                 clip_model_path: str = "openai/clip-vit-large-patch14",
                 t5_model_path: str = "google/t5-v1_1-xxl", 
                 text_encoder_quantization: str = "none",
                 clip_skip: int = 0,
                 device_auto_detect: bool = True):
        """
        Initialize FluxTextEncoders node.
        
        Args:
            gpu_device: GPU device ID to use for text encoders
            memory_fraction: Fraction of GPU memory to allocate (0.1-1.0)
            clip_model_path: Path or HuggingFace model ID for CLIP encoder
            t5_model_path: Path or HuggingFace model ID for T5 encoder
            text_encoder_quantization: Quantization type ("none", "4bit", "8bit")
            clip_skip: Number of CLIP layers to skip from the end
            device_auto_detect: Whether to auto-detect optimal GPU device
        """
        self.gpu_device = gpu_device
        self.memory_fraction = memory_fraction
        self.clip_model_path = clip_model_path
        self.t5_model_path = t5_model_path
        self.text_encoder_quantization = text_encoder_quantization
        self.clip_skip = clip_skip
        self.device_auto_detect = device_auto_detect
        
        # Model components
        self.clip_tokenizer = None
        self.clip_text_encoder = None
        self.t5_tokenizer = None
        self.t5_text_encoder = None
        
        # Device and memory management
        self.device = None
        self.is_initialized = False
        
        # Available GPUs for parameter validation
        self.available_gpus = get_available_gpus()
        
    def get_gpu_choices(self) -> List[tuple]:
        """Get available GPU choices for UI dropdown."""
        choices = []
        for gpu in self.available_gpus:
            if gpu['is_available']:
                name = f"GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)"
                choices.append((name, gpu['id']))
        
        if not choices:
            choices = [("No CUDA GPUs Available", -1)]
            
        return choices
    
    def setup_device(self) -> bool:
        """Setup GPU device and memory allocation."""
        try:
            if not self.available_gpus:
                logger.error("No CUDA GPUs available")
                return False
                
            # Auto-detect optimal device if enabled
            if self.device_auto_detect:
                # Estimate memory requirement for text encoders (roughly 6-8GB for T5-XXL)
                required_memory = 8.0
                optimal_device = None
                
                for gpu in self.available_gpus:
                    if gpu['is_available'] and gpu['memory_gb'] * self.memory_fraction >= required_memory:
                        optimal_device = gpu['id']
                        break
                        
                if optimal_device is not None:
                    self.gpu_device = optimal_device
                    logger.info(f"Auto-detected optimal device: GPU {optimal_device}")
            
            # Select and configure GPU
            if not select_gpu(self.gpu_device, self.memory_fraction):
                logger.error(f"Failed to select GPU {self.gpu_device}")
                return False
                
            self.device = torch.device(f"cuda:{self.gpu_device}")
            logger.info(f"Using device: {self.device} with {self.memory_fraction*100:.0f}% memory")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up device: {e}")
            return False
    
    def load_text_encoders(self) -> bool:
        """Load CLIP and T5 text encoders with optional quantization."""
        try:
            # Import required libraries
            transformers = import_optional_dependency("transformers")
            CLIPTextModel = transformers.CLIPTextModel
            CLIPTokenizer = transformers.CLIPTokenizer
            T5EncoderModel = transformers.T5EncoderModel
            T5TokenizerFast = transformers.T5TokenizerFast
            
            # Setup quantization config if needed
            quantization_config = None
            if self.text_encoder_quantization in ["4bit", "8bit"]:
                try:
                    from transformers import BitsAndBytesConfig
                    
                    if self.text_encoder_quantization == "4bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16
                        )
                    elif self.text_encoder_quantization == "8bit":
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        
                    logger.info(f"Using {self.text_encoder_quantization} quantization")
                    
                except ImportError:
                    logger.warning("bitsandbytes not available, using full precision")
                    quantization_config = None
            
            # Load CLIP text encoder
            logger.info(f"Loading CLIP text encoder: {self.clip_model_path}")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_path)
            self.clip_text_encoder = CLIPTextModel.from_pretrained(
                self.clip_model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if quantization_config is None else None,
                device_map=self.device if quantization_config is None else "auto"
            )
            
            if quantization_config is None:
                self.clip_text_encoder = self.clip_text_encoder.to(self.device)
            
            # Load T5 text encoder  
            logger.info(f"Loading T5 text encoder: {self.t5_model_path}")
            self.t5_tokenizer = T5TokenizerFast.from_pretrained(self.t5_model_path)
            self.t5_text_encoder = T5EncoderModel.from_pretrained(
                self.t5_model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if quantization_config is None else None,
                device_map=self.device if quantization_config is None else "auto"
            )
            
            if quantization_config is None:
                self.t5_text_encoder = self.t5_text_encoder.to(self.device)
                
            logger.info("Text encoders loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading text encoders: {e}")
            return False
    
    def encode_prompts(self, 
                      prompts: Union[str, List[str]], 
                      prompt_weights: Optional[List[float]] = None) -> Optional[TextEmbeddingArtifact]:
        """
        Encode prompts using CLIP and T5 text encoders.
        
        Args:
            prompts: Single prompt string or list of prompts
            prompt_weights: Optional weights for combining multiple prompts
            
        Returns:
            TextEmbeddingArtifact containing the encoded embeddings
        """
        try:
            # Ensure prompts is a list
            if isinstance(prompts, str):
                prompts = [prompts]
                
            # Validate prompt weights
            if prompt_weights is not None:
                if len(prompt_weights) != len(prompts):
                    logger.error("Number of prompt weights must match number of prompts")
                    return None
            else:
                prompt_weights = [1.0] * len(prompts)
            
            # Encode with CLIP
            clip_inputs = self.clip_tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=77
            ).to(self.device)
            
            with torch.no_grad():
                if self.clip_skip > 0:
                    # Use hidden states from earlier layer if clip_skip is specified
                    clip_outputs = self.clip_text_encoder(**clip_inputs, output_hidden_states=True)
                    pooled_prompt_embeds = clip_outputs.hidden_states[-(self.clip_skip + 1)]
                    pooled_prompt_embeds = self.clip_text_encoder.text_model.final_layer_norm(pooled_prompt_embeds)
                else:
                    clip_outputs = self.clip_text_encoder(**clip_inputs)
                    pooled_prompt_embeds = clip_outputs.pooler_output
            
            # Encode with T5
            t5_inputs = self.t5_tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                t5_outputs = self.t5_text_encoder(**t5_inputs)
                prompt_embeds = t5_outputs.last_hidden_state
            
            # Apply prompt weights if multiple prompts
            if len(prompts) > 1:
                # Weighted average of embeddings
                weights_tensor = torch.tensor(prompt_weights, device=self.device, dtype=prompt_embeds.dtype)
                weights_tensor = weights_tensor / weights_tensor.sum()
                
                prompt_embeds = torch.sum(prompt_embeds * weights_tensor.view(-1, 1, 1), dim=0, keepdim=True)
                pooled_prompt_embeds = torch.sum(pooled_prompt_embeds * weights_tensor.view(-1, 1), dim=0, keepdim=True)
            
            # Create artifact
            artifact = TextEmbeddingArtifact(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                original_prompts=prompts,
                prompt_weights=prompt_weights,
                device=str(self.device),
                dtype=str(prompt_embeds.dtype)
            )
            
            logger.info(f"Successfully encoded {len(prompts)} prompt(s)")
            logger.info(f"Prompt embeds shape: {prompt_embeds.shape}")
            logger.info(f"Pooled embeds shape: {pooled_prompt_embeds.shape}")
            
            return artifact
            
        except Exception as e:
            logger.error(f"Error encoding prompts: {e}")
            return None
    
    def run(self, 
            prompt: str = "",
            prompts: Optional[List[str]] = None,
            prompt_weights: Optional[List[float]] = None,
            **kwargs) -> Optional[TextEmbeddingArtifact]:
        """
        Main execution method for the node.
        
        Args:
            prompt: Single prompt string (used if prompts is None)
            prompts: List of prompts for multi-prompt encoding
            prompt_weights: Optional weights for combining multiple prompts
            **kwargs: Additional arguments
            
        Returns:
            TextEmbeddingArtifact or None if failed
        """
        try:
            # Initialize if not already done
            if not self.is_initialized:
                if not self.setup_device():
                    return None
                    
                if not self.load_text_encoders():
                    return None
                    
                self.is_initialized = True
            
            # Determine which prompts to use
            if prompts is not None and len(prompts) > 0:
                input_prompts = prompts
            elif prompt:
                input_prompts = [prompt]
            else:
                logger.error("No prompts provided")
                return None
            
            # Encode prompts
            result = self.encode_prompts(input_prompts, prompt_weights)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in FluxTextEncoders.run(): {e}")
            return None
    
    def cleanup(self):
        """Clean up GPU memory and model resources."""
        try:
            # Clear models
            if self.clip_text_encoder is not None:
                del self.clip_text_encoder
                self.clip_text_encoder = None
                
            if self.t5_text_encoder is not None:
                del self.t5_text_encoder
                self.t5_text_encoder = None
                
            if self.clip_tokenizer is not None:
                del self.clip_tokenizer
                self.clip_tokenizer = None
                
            if self.t5_tokenizer is not None:
                del self.t5_tokenizer
                self.t5_tokenizer = None
            
            # Clean up GPU memory
            cleanup_gpu_memory(self.gpu_device)
            
            self.is_initialized = False
            logger.info("FluxTextEncoders cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup.""" 
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction


if __name__ == "__main__":
    # Test the FluxTextEncoders node
    print("=== FluxTextEncoders Test ===")
    
    # Initialize node
    text_encoders = FluxTextEncoders(
        gpu_device=0,
        memory_fraction=0.4,
        device_auto_detect=True
    )
    
    print(f"Available GPUs: {len(text_encoders.available_gpus)}")
    for gpu in text_encoders.available_gpus:
        print(f"  {gpu['name']}: {gpu['memory_gb']:.1f}GB")
    
    # Test prompt encoding (will need transformers installed)
    try:
        result = text_encoders.run(prompt="A cat in a spacesuit floating in space")
        
        if result:
            print(f"✅ Successfully encoded prompt")
            print(f"   Prompt embeds shape: {result.prompt_embeds.shape}")
            print(f"   Pooled embeds shape: {result.pooled_prompt_embeds.shape}")
            print(f"   Device: {result.device}")
        else:
            print("❌ Failed to encode prompt")
            
    except ImportError:
        print("⚠️  transformers not installed - skipping encoding test")
    except Exception as e:
        print(f"❌ Error during encoding test: {e}")
    
    # Cleanup
    text_encoders.cleanup() 