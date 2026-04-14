"""
configuration
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DreamBoothConfig:
    # model
    pretrained_model: str = "sd-legacy/stable-diffusion-v1-5"
    revision: Optional[str] = None

    # subject
    # unique identifier token
    identifier_token: str = "sks"
    class_noun: str = "cat"

    # data
    subject_dir: str = "./data/sompong_images"         
    class_images_dir: str = "./data/class_images" 
    output_dir: str = "./output"

    # hyperparameters
    learning_rate: float = 5e-6
    max_train_steps: int = 1000
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1

    # AdamW
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # prior preservation
    prior_preservation: bool = True
    prior_loss_weight: float = 1.0
    num_class_images: int = 64

    # image
    resolution: int = 512
    center_crop: bool = True

    # finetune
    train_text_encoder: bool = True

    # inference
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    seed: int = 42
    mixed_precision: str = "fp16"
    save_every_n_steps: int = 500
    log_every_n_steps: int = 50

    @property
    def instance_prompt(self) -> str:
        return f"a {self.identifier_token} {self.class_noun}"

    @property
    def class_prompt(self) -> str:
        return f"a {self.class_noun}"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.class_images_dir).mkdir(parents=True, exist_ok=True)