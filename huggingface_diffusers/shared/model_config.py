from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Minimal model selection for Diffusers FLUX.
    Stored fields mirror the bkup selector but used as a plain dict across nodes.
    """
    model_id: str
    clip_id: str
    t5_id: str
    variant: str = "dev"
    quantization: str = "none"  # none|8-bit (transformer stays fp16/bf16)
    local_path: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelConfig":
        return ModelConfig(
            model_id=data.get("model_id", "black-forest-labs/FLUX.1-dev"),
            clip_id=data.get("clip_id", "openai/clip-vit-large-patch14"),
            t5_id=data.get("t5_id", "google/t5-v1_1-xxl"),
            variant=data.get("variant", "dev"),
            quantization=data.get("quantization", "none"),
            local_path=data.get("local_path"),
        )



