"""
Each training batch contains:
  1. a subject image paired with the instance prompt  (ex:"a [V] [class noun]")
  2. a class image paired with "a [class noun]"
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DreamBoothDataset(Dataset):
    """
    Dataset that serves (image, prompt) pairs for training.

    When prior_preservation=True, each sample is a dict with both:
      -instance_image/instance_prompt  (the subject)
      -class_image/class_prompt   (the prior preservation data)
    """

    def __init__(
        self,
        subject_dir: str,
        instance_prompt: str,
        tokenizer,
        class_images_dir: str | None = None,
        class_prompt: str | None = None,
        resolution: int = 512,
        center_crop: bool = True,
    ):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.resolution = resolution

        # load instance images
        subject_path = Path(subject_dir)
        self.instance_images = sorted([
            p for p in subject_path.iterdir()
            if p.suffix.lower() in {".jpeg"}
        ])
        if len(self.instance_images) == 0:
            raise ValueError(f"No images found in {subject_dir}")
        print(f"[Dataset] Found {len(self.instance_images)} subject images")

        # class images
        self.class_images = []
        if class_images_dir and class_prompt:
            class_path = Path(class_images_dir)
            self.class_images = sorted([
                p for p in class_path.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ])
            print(f"[Dataset] Found {len(self.class_images)} class prior images")

        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR), # match the shorter side of the images
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution), # size: resolution * resolution
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # normalize pixel values to [-1, 1]
        ])

    @property
    
    def has_class_images(self) -> bool:
        """
        Return True if class images are available for prior preservation
        """
        return len(self.class_images) > 0

    def __len__(self):
        """ Return dataset length. If class images are used, return the length of the longer set"""
        if self.has_class_images:
            return max(len(self.instance_images), len(self.class_images))
        return len(self.instance_images)
    
    def _tokenize(self, prompt: str):
        """ Convert a prompt string into token IDs for the text encoder"""
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0) 

    def __getitem__(self, index: int) -> dict:
        """ Return one training sample.
            Each sample includes:
            - a subject image with its prompt
            - (optional) a class image with its prompt.
            Indexing wraps around if one set is smaller than the other. 
        """
        
        sample = {}

        # instance img
        instance_img_path = self.instance_images[index % len(self.instance_images)]
        instance_image = Image.open(instance_img_path).convert("RGB")
        sample["instance_image"] = self.image_transforms(instance_image)
        sample["instance_prompt_ids"] = self._tokenize(self.instance_prompt)

        # class img
        if self.has_class_images:
            class_img_path = self.class_images[index % len(self.class_images)]
            class_image = Image.open(class_img_path).convert("RGB")
            sample["class_image"] = self.image_transforms(class_image)
            sample["class_prompt_ids"] = self._tokenize(self.class_prompt)

        return sample
    
    def combine(batch: list[dict], prior_preservation: bool = False) -> dict:
        """
        Stacks instance and class data.
    
        When prior_preservation=True, instance and class images are concatenated
        along the batch dimension so they go through the model together.
        """
        instance_images = torch.stack([s["instance_image"] for s in batch])
        instance_prompt_ids = torch.stack([s["instance_prompt_ids"] for s in batch])
    
        if prior_preservation and "class_image" in batch[0]:
            class_images = torch.stack([s["class_image"] for s in batch])
            class_prompt_ids = torch.stack([s["class_prompt_ids"] for s in batch])
    
            # first half of batch = instance, second half = class
            images = torch.cat([instance_images, class_images], dim=0)
            prompt_ids = torch.cat([instance_prompt_ids, class_prompt_ids], dim=0)
        else:
            images = instance_images
            prompt_ids = instance_prompt_ids
    
        return {"pixel_values": images, "input_ids": prompt_ids}