# DreamBooth Reimplementation

## 1. Introduction

This repository is a CS4782 Spring 2026 final project that attempts to reimplement the paper **DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**.

DreamBooth's main contribution is few-shot personalization of a pretrained text-to-image diffusion model: given 3-5 images, the model learns to bind a unique identifier token for a subject while retaining the class prior needed to place that subject in new contexts.

## 2. Chosen Result

We tried to reproduce DreamBooth's **subject-driven recontextualization** result: generating a specific subject in new scenes while preserving its visual identity.

This corresponds to the qualitative comparison in **Figure 4** and the quantitative metric comparison in **Table 1** from the original paper.

**Figure 4: Qualitative subject-driven recontextualization comparison**

<img width="449" height="527" alt="Figure 4: Qualitative subject-driven recontextualization comparison" src="https://github.com/user-attachments/assets/aee2d273-5176-4434-b629-f8ef0c68e33a" />

**Table 1: Quantitative comparison using DINO, CLIP-I, and CLIP-T**

<img width="441" height="173" alt="Table 1: Quantitative comparison using DINO, CLIP-I, and CLIP-T" src="https://github.com/user-attachments/assets/915c0890-49eb-4a78-9088-e9517feb5476" />


## 3. GitHub Contents

- `code/`: Scripts for DreamBooth dataset preparation, class-prior image generation, training, inference, token utilities, and performance evaluation.

- `data/`: Subject images, generated class-prior images, and saved metric logs.

- `results/`: Generated inference images and evaluation outputs, including:
  - Images generated with prior preservation loss weights of 0, 0.25, 0.5, 0.75, and 1.0 at 500 training steps.
  - Images generated with prior preservation loss weight of 0 at 600, 650, 700, and 800 training steps.
  - Saved model for prior loss weights of 0 and 0.25 at 500 training steps.
  - Other images generated with different prompts/contexts.
  - Evaluation table comparing DINO, CLIP-I, and CLIP-T scores.

- `poster/`: PDF version of the final DreamBooth reimplementation poster.

- `report/`: PDF version of the final DreamBooth reimplementation report.



## 4. Re-implementation Details

We fine-tuned `sd-legacy/stable-diffusion-v1-5` on 5 subject images using the identifier `vy` and class noun `water bottle`, with 200 generated water-bottle class-prior images.

Training uses 512x512 crops, batch size 1, learning rate `5e-6`, fp16 CUDA execution, 8-bit Adam, identifier-embedding training, cross-attention UNet updates, and generated class-prior batches.

Evaluation uses the paper's metrics: **CLIP-I** and **DINO** for subject fidelity, plus **CLIP-T** for prompt fidelity.

Main modifications from the original paper: Our implementation uses Stable Diffusion v1.5 instead of Imagen and fine-tunes only the identifier token embedding plus selected UNet cross-attention layers to fit Colab GPU limits. We also scale down the experiment to one subject category, water bottle, using 5 subject images, 200 generated class-prior images, and a smaller prompt set. Because of these constraints, our results focus on small-scale subject binding and metric comparison rather than full reproduction of the paper’s large-scale results

## 5. Reproduction Steps


Install dependencies in a CUDA environment:

```bash
pip install torch torchvision diffusers transformers accelerate pillow tqdm bitsandbytes
```
Here, replace `<identifier>` with your unique token, `<class noun>` with the subject class, `<class_name>` with a folder-safe version of the class name, and `<new context>` with the inference setting you want to test.



Generate class-prior images:

```bash
python code/generate_class_data.py \
  --class_prompt "a photo of a <class noun>" \
  --num_images 200 \
  --output_dir data/class_images_<class_name> \
  --batch_size 4
```

Train the personalized model:

```bash
python code/train.py \
  --subject_dir data/<subject_images_dir> \
  --class_dir data/class_images_<class_name> \
  --class_noun "<class noun>" \
  --identifier "<identifier>" \
  --output_dir output/models/model-<identifier>-<class_name> \
  --steps 800 \
  --lr 5e-6 \
  --unet_train_mode cross_attention
```

Generate images:

```bash
python code/inference.py \
  --model_path output/models/model-<identifier>-<class_name>/final \
  --prompt "a photo of a <identifier> <class noun> in <new context>" \
  --num_images 4 \
  --output_dir output/generated_<class_name>
```

Evaluate generated images:

```bash
python code/performance.py \
  --real_dir data/<subject_images_dir> \
  --gen_dir output/generated_<class_name> \
  --prompt "a photo of a <identifier> <class noun> in <new context>" \
  --output data/clip_dino_outputs/<run_name>
```

We used T4 GPU on Colab during training.

## 6. Results / Insights
### Expected Results

Running this repository trains a small-scale DreamBooth-style personalized Stable Diffusion model. The final output is a model that can generate the target subject in new contexts using prompts that include the chosen identifier token.

The repository also produces generated images and evaluation outputs using DINO, CLIP-I, and CLIP-T. These metrics compare subject fidelity and prompt fidelity against the evaluation style used in the original DreamBooth paper.

You can expect reasonable subject-conditioned generation and metric logs, but not full reproduction of the original paper’s large-scale performance.

### Our Result and Insights

<img width="406" height="88" alt="Evaluation table comparing DINO, CLIP-I, and CLIP-T scores" src="https://github.com/user-attachments/assets/0bebcf14-8f65-448b-b789-d10a027cde50" />


Our CLIP-I and CLIP-T scores are comparable to the paper's Stable Diffusion reference on this single-subject run, but DINO is lower, suggesting that the preservation of some fine-grained details is weaker.

## 7. Conclusion

This reimplementation shows that the core DreamBooth idea can be reproduced with Stable Diffusion v1.5 under limited GPU resources.

The main lesson is that prompt fidelity is achievable with a compact setup, while robust subject identity and diversity require careful tuning of training length, trainable model components, and prior preservation. In our experiments, prior preservation loss did not appear to be essential for achieving reasonable subject-conditioned generation. This suggests that, under a partial fine-tuning setup, prior preservation may be less important than in the original full fine-tuning DreamBooth setting.

## 8. References

- Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman. [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242). CVPR 2023.
- Hugging Face Diffusers. [DreamBooth training guide](https://huggingface.co/docs/diffusers/v0.15.0/en/training/dreambooth).
- Hugging Face. [Stable Diffusion v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
- OpenAI. [CLIP](https://github.com/openai/CLIP).
- Facebook Research. [DINO](https://github.com/facebookresearch/dino).

## 9. Acknowledgements

This project was developed as part of CS4782 Spring 2026 coursework.
