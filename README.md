# DreamBooth Reimplementation

## 1. Introduction

This repository is a CS4782 Spring 2026 final project that attempts to reimplement the paper **DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**.

DreamBooth's main contribution is few-shot personalization of a pretrained text-to-image diffusion model: given 3-5 images, the model learns to bind a unique identifier token for a subject while retaining the class prior needed to place that subject in new contexts.

## 2. Chosen Result

We tried to reproduce DreamBooth's **subject-driven recontextualization** result to generate a specific subject in new scenes while preserving its visual identity.

This corresponds to qualitative results in **Figure 4** and the quantitative metrics in **Table 1** in the paper.

// UPLOAD THE IMAGES

## 3. GitHub Contents

- `code/`: DreamBooth dataset, class-prior generation, training, inference, token utilities, and performance evaluation scripts.
- `data/`: subject images, generated class-prior images, and saved metric logs
- `results/`:

• code/: A directory containing your re-implementation code, along with any necessary configuration
files or scripts.
• data/: A directory containing the datasets used for training and evaluation, or a README with
instruction on how to obtain the dataset.
• results/: A directory containing the results of your re-implementation, including any generated figures,
tables, or log files.
• poster/: A directory containing a PDF of the poster used for your in-class presentations.
• report/: A directory containing a PDF of the final report submitted.
• LICENSE: A file specifying the license under which your code is released (e.g., MIT, Apache 2.0).
• .gitignore: A file specifying files or directories that should be ignored by Git.

## 4. Re-implementation Details

The implementation fine-tunes `sd-legacy/stable-diffusion-v1-5` on 5 subject images using the identifier `vy` and class noun `water bottle`, with 200 generated water-bottle class-prior images.

Training uses 512x512 crops, batch size 1, learning rate `5e-6`, fp16 CUDA execution, 8-bit Adam, identifier-embedding training, cross-attention UNet updates, and generated class-prior batches.

Evaluation uses the paper's metrics: **CLIP-I** and **DINO** for subject fidelity, plus **CLIP-T** for prompt fidelity.

Main modifications from the original paper: Stable Diffusion v1.5 replaces Imagen, only a small number of subjects/prompts are tested, and partial UNet fine-tuning is used to fit Colab-scale GPU memory.

## 5. Reproduction Steps

Install dependencies in a CUDA environment:

```bash
pip install torch torchvision diffusers transformers accelerate pillow tqdm bitsandbytes
```

Generate class-prior images:

```bash
python code/generate_class_data.py \
  --class_prompt "a photo of a water bottle" \
  --num_images 200 \
  --output_dir data/class_images_waterbottle \
  --batch_size 4
```

Train the personalized model:

```bash
python code/train.py \
  --subject_dir data/water-bottle-subject-img \
  --class_dir data/class_images_waterbottle \
  --class_noun "water bottle" \
  --identifier "vy" \
  --output_dir output/models/model-vy-water-bottle \
  --steps 800 \
  --lr 5e-6 \
  --unet_train_mode cross_attention
```

Generate images:

```bash
python code/inference.py \
  --model_path output/models/model-vy-water-bottle/final \
  --prompt "a photo of a vy water bottle on a beach" \
  --num_images 4 \
  --output_dir output/generated_water_bottle
```

Evaluate generated images:

```bash
python code/performance.py \
  --real_dir data/water-bottle-subject-img \
  --gen_dir output/generated_water_bottle \
  --prompt "a photo of a vy water bottle on a beach" \
  --output data/clip_dino_outputs/my_run
```

A CUDA GPU is strongly recommended; the low-memory configuration targets roughly 14-16 GB VRAM.

## 6. Results / Insights

Best committed water-bottle beach run:

| Run                                             |   DINO | CLIP-I | CLIP-T |
| ----------------------------------------------- | -----: | -----: | -----: |
| DreamBooth Stable Diffusion, paper Table 1      |  0.668 |  0.803 |  0.305 |
| Ours, water bottle, prior weight 0.0, 800 steps | 0.5535 | 0.8263 | 0.3127 |
| Ours, water bottle, prior weight 0.0, 500 steps | 0.5472 | 0.7661 | 0.3171 |

Our CLIP-I and CLIP-T scores are comparable to the paper's Stable Diffusion reference on this single-subject run, but DINO is lower, suggesting weaker fine-grained subject identity preservation.

This is not an apples-to-apples benchmark: the paper's Table 1 averages over 30 subjects and many prompts, while this repository reports a focused water-bottle reproduction.

## 7. Conclusion

This reimplementation shows that the core DreamBooth idea can be reproduced with Stable Diffusion v1.5 under limited GPU resources.

The main lesson is that prompt fidelity is achievable with a compact setup, while robust subject identity requires careful tuning of prior preservation, training length, and trainable model components.

## 8. References

- Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman. [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242). CVPR 2023.
- Hugging Face Diffusers. [DreamBooth training guide](https://huggingface.co/docs/diffusers/v0.15.0/en/training/dreambooth).
- Hugging Face. [Stable Diffusion v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
- OpenAI. [CLIP](https://github.com/openai/CLIP).
- Facebook Research. [DINO](https://github.com/facebookresearch/dino).

## 9. Acknowledgements

This project was developed as part of CS4782 Spring 2026 coursework.
