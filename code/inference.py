"""
generate new images of the fine-tuned subject given text prompts.
ex:
  "a [V] dog on the beach"
  "a painting of a [V] dog in the style of Van Gogh"
  "a [V] dog wearing a chef hat"
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def generate(
    model_path: str,
    prompt: str,
    num_images: int = 4,
    output_dir: str = "./generated",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 42,
    negative_prompt: str | None = None,
):
    """
    generate images using the already fine-tuned model

    Args:
        model_path: Path to the saved fine-tuned model
        prompt: Text prompt containing the unique identifier
                e.g. "a sks dog on a surfboard"
        num_images: Number of images to generate
        output_dir: Where to save generated images
        guidance_scale: Classifier-free guidance scale (higher = more prompt-faithful)
        num_inference_steps: Number of denoising steps
        seed: Random seed
        negative_prompt: Optional negative prompt for better quality
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # push the generated image away from the negative prompt
    if negative_prompt is None:
        negative_prompt = "blurry, low quality, distorted, deformed"

    print(f"Generating {num_images} images for: '{prompt}'")

    for i in range(num_images):
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # create a file for each image i
        safe_prompt = prompt.replace(" ", "_")[:50]
        filename = f"{safe_prompt}_{i:02d}.png"
        image.save(output_path / filename)
        print(f"  Saved: {filename}")

    print(f"\nAll images saved to {output_dir}")


def generate_grid(
    model_path: str,
    prompts: list[str],
    output_path: str = "./generated/grid.png",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 42,
):
    """
    generate a grid of images from multiple prompts (just for convenience)
    """
    from PIL import Image

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = []

    for prompt in prompts:
        img = pipeline(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        images.append(img)
        print(f"  Generated: {prompt}")

    # now we actually assemble into a grid
    n = len(images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"\nGrid saved to {output_path}")


# # **** example prompts here for convenience! we can use these to test
 
EXAMPLE_PROMPTS = {
    "recontextualization": [
        "a {id} {cls} in the jungle",
        "a {id} {cls} in the snow",
        "a {id} {cls} on the beach",
        "a {id} {cls} on a cobblestone street",
        "a {id} {cls} with the Eiffel Tower in the background",
        "a {id} {cls} floating on top of water",
        "a {id} {cls} with a mountain in the background",
    ],
    "art_renditions": [
        "a painting of a {id} {cls} in the style of Vincent Van Gogh",
        "a painting of a {id} {cls} in the style of Leonardo da Vinci",
        "a statue of a {id} {cls} in the style of Michelangelo",
        "a watercolor painting of a {id} {cls}",
        "an oil painting of a {id} {cls}",
    ],
    "property_modification": [
        "a red {id} {cls}",
        "a purple {id} {cls}",
        "a shiny {id} {cls}",
        "a transparent {id} {cls}",
        "a wet {id} {cls}",
    ],
    "view_synthesis": [
        "a {id} {cls} seen from the top",
        "a {id} {cls} seen from the bottom",
        "a {id} {cls} seen from the side",
        "a {id} {cls} seen from the back",
    ],
    "accessorization": [
        "a {id} {cls} wearing a red hat",
        "a {id} {cls} wearing a santa hat",
        "a {id} {cls} in a chef outfit",
        "a {id} {cls} in a police outfit",
        "a {id} {cls} wearing pink glasses",
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with DreamBooth model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--prompt", type=str, default=None,
                        help='Generation prompt, e.g. "a sks dog on the beach"')
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="./generated",
                        help="Output directory for generated images")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # (for grid generation with example prompts)
    parser.add_argument("--grid", action="store_true",
                        help="Generate a grid using example prompts")
    parser.add_argument("--category", type=str, default="recontextualization",
                        choices=list(EXAMPLE_PROMPTS.keys()),
                        help="Category of example prompts for grid mode")
    parser.add_argument("--identifier", type=str, default="sks")
    parser.add_argument("--class_noun", type=str, default="dog")

    args = parser.parse_args()

    if args.grid:
        prompts = [
            p.format(id=args.identifier, cls=args.class_noun)
            for p in EXAMPLE_PROMPTS[args.category]
        ]
        generate_grid(
            model_path=args.model_path,
            prompts=prompts,
            output_path=f"{args.output_dir}/grid_{args.category}.png",
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
    elif args.prompt:
        generate(
            model_path=args.model_path,
            prompt=args.prompt,
            num_images=args.num_images,
            output_dir=args.output_dir,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
    else:
        print("Provide --prompt or use --grid mode. See --help for details.")
