"""
Evaluate the generated images using CLIP-I, CLIP-T, and DINO.

Run:
    python performance.py --real_dir <path_to_real_images> \
                                  --gen_dir <path_to_generated_images> \
                                  --prompt "<text prompt used for generation>"

Requirements:
    pip install torch torchvision transformers pillow
"""

import argparse
import os
import glob
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as T


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_image_paths(directory: str) -> list[str]:
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return paths

def load_clip(device: torch.device):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor

def get_clip_image_embedding(image_path: str, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb / emb.norm(dim=-1, keepdim=True)

def compute_clip_i(real_paths, gen_paths, model, processor, device) -> float:
    real_embs = [get_clip_image_embedding(p, model, processor, device) for p in real_paths]
    gen_embs = [get_clip_image_embedding(p, model, processor, device) for p in gen_paths]

    scores = []
    for g in gen_embs:
        for r in real_embs:
            scores.append((g @ r.T).item())
    return sum(scores) / len(scores) # average score

def compute_clip_t(gen_paths, prompt: str, model, processor, device) -> float:
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    scores = []
    for p in gen_paths:
        img_emb = get_clip_image_embedding(p, model, processor, device)
        scores.append((img_emb @ text_emb.T).item())
    return sum(scores) / len(scores)

def load_dino(device: torch.device):
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    model = model.to(device).eval()
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

def get_dino_embedding(image_path: str, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    return emb / emb.norm(dim=-1, keepdim=True)


def compute_dino(real_paths, gen_paths, model, transform, device) -> float:
    real_embs = [get_dino_embedding(p, model, transform, device) for p in real_paths]
    gen_embs = [get_dino_embedding(p, model, transform, device) for p in gen_paths]

    scores = []
    for g in gen_embs:
        for r in real_embs:
            scores.append((g @ r.T).item())
    return sum(scores) / len(scores)

def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIP-I, CLIP-T, and DINO evaluation metrics for subject-driven generation."
    )
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Path to folder of real/reference subject images.")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Path to folder of generated subject images.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt used to generate the images.")
    parser.add_argument("--output", type=str, default="performance.txt",
                        help="Output file for results (default: performance.txt).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    real_paths = collect_image_paths(args.real_dir)
    gen_paths = collect_image_paths(args.gen_dir)
    print(f"Found {len(real_paths)} real images and {len(gen_paths)} generated images.")

    print("Loading CLIP model...")
    clip_model, clip_processor = load_clip(device)

    print("Computing CLIP-I (subject fidelity)...")
    clip_i = compute_clip_i(real_paths, gen_paths, clip_model, clip_processor, device)

    print("Computing CLIP-T (prompt fidelity)...")
    clip_t = compute_clip_t(gen_paths, args.prompt, clip_model, clip_processor, device)

    # Free CLIP memory before loading DINO
    del clip_model, clip_processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Loading DINO model...")
    dino_model, dino_transform = load_dino(device)

    print("Computing DINO (subject fidelity)...")
    dino = compute_dino(real_paths, gen_paths, dino_model, dino_transform, device)

    # write the results
    output_path = Path(args.output)
    lines = [
        "DreamBooth Evaluation Metrics",
        "=" * 40,
        f"Real image directory : {args.real_dir}",
        f"Generated image directory : {args.gen_dir}",
        f"Prompt : {args.prompt}",
        f"Number of real images : {len(real_paths)}",
        f"Number of generated images : {len(gen_paths)}",
        "",
        "Results",
        "-" * 40,
        f"CLIP-I (subject fidelity) : {clip_i:.4f}",
        f"DINO   (subject fidelity) : {dino:.4f}",
        f"CLIP-T (prompt fidelity)  : {clip_t:.4f}",
        "",
        "Reference ranges (from DreamBooth paper)",
        "-" * 40,
        "Real images upper bound : DINO ~0.774, CLIP-I ~0.885",
        "DreamBooth (Imagen)     : DINO ~0.696, CLIP-I ~0.812, CLIP-T ~0.306",
        "DreamBooth (SD)         : DINO ~0.668, CLIP-I ~0.803, CLIP-T ~0.305",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()

