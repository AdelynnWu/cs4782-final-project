"""
Generate Class Prior Images
"""
 
import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


def generate_class_images(
    class_prompt: str,
    num_images: int,
    output_dir: str,
    model_name: str = "sd-legacy/stable-diffusion-v1-5",
    batch_size: int = 4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    seed: int = 42,
):
    """
    Generate class prior images using the frozen pretrained model.

    Args:
        class_prompt: e.g. "a cat" — the class-level prompt without the identifier
        num_images: 200-1000
        output_dir: Where to save generated images
        model_name: Pretrained model to use
        batch_size: Images per forward pass
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Denoising steps
        seed: Random seed for reproducibility
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = list(output_path.glob("*.png"))
    if len(existing) >= num_images:
        print(f"Already have {len(existing)} class images in {output_dir}, skipping.")
        return

    remaining = num_images - len(existing)
    start_idx = len(existing)
    print(f"Generating {remaining} class images with prompt: '{class_prompt}'")

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # ── Generate in batches ───────────────────────────────────────────
    num_batches = (remaining + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating class images"):
        current_batch_size = min(batch_size, remaining - batch_idx * batch_size)

        images = pipeline(
            prompt=[class_prompt] * current_batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        for i, img in enumerate(images):
            img_idx = start_idx + batch_idx * batch_size + i
            img.save(output_path / f"class_{img_idx:04d}.png")

    print(f"Done. {num_images} class images saved to {output_dir}")

    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate class prior images for DreamBooth")
    parser.add_argument("--class_prompt", type=str, required=True,
                        help='Class-level prompt, e.g. "a cat"')
    parser.add_argument("--num_images", type=int, default=200,
                        help="Number of images to generate (default: 200)")
    parser.add_argument("--output_dir", type=str, default="./data/class_images",
                        help="Output directory for generated images")
    parser.add_argument("--model_name", type=str,
                        default="sd-legacy/stable-diffusion-v1-5",
                        help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    generate_class_images(
        class_prompt=args.class_prompt,
        num_images=args.num_images,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        seed=args.seed,
    )