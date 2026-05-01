"""
Training loop
"""

import argparse
import math
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from config import DreamBoothConfig
from dataset import DreamBoothDataset

def load_model_parts(config, device, weight_dtype):
    """load tokenizer, text encoder, VAE, UNet, and scheduler"""
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    # load in fp32; train() will place on device with correct dtype
    unet.train()

    if config.train_text_encoder:
        text_encoder.train()
    else:
        text_encoder.requires_grad_(False)

    return tokenizer, text_encoder, vae, unet, noise_scheduler

def train(config: DreamBoothConfig):
    """Run the full DreamBooth training pipeline."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    print("=" * 60)
    print("DreamBooth Training")
    print("=" * 60)
    print(f"  Instance prompt: '{config.instance_prompt}'")
    print(f"  Class prompt:    '{config.class_prompt}'")
    print(f"  Prior preservation: {config.prior_preservation} (λ={config.prior_loss_weight})")
    print(f"  Learning rate:   {config.learning_rate}")
    print(f"  Max steps:       {config.max_train_steps}")
    print(f"  Train text encoder: {config.train_text_encoder}")
    print(f"  Device: {device}, dtype: {weight_dtype}")
    print("=" * 60)


    # LOAD PRETRAINED MODEL COMPONENTS

    print("\n[1/5] Loading pretrained model components...")
    # load model 
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_model_parts(
    config,device, weight_dtype)

    # freeze VAE — fp16 is fine since it is never updated
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    # keep UNet in fp32 so Adam optimizer states don't overflow;
    # autocast handles fp16 compute in the forward pass
    unet.to(device, dtype=torch.float32)
    unet.train()
    # gradient checkpointing recomputes activations during backward instead of
    # storing them — cuts activation memory ~60% at the cost of ~25% more compute
    unet.enable_gradient_checkpointing()

    # freeze text encoder to save ~1 GB of model + optimizer memory on Colab;
    # fp16 is fine since its weights are never updated
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)
    if config.train_text_encoder:
        text_encoder.to(device, dtype=torch.float16)
        text_encoder.requires_grad_(True)
        text_encoder.train()
        text_encoder.gradient_checkpointing_enable()
    else:
        text_encoder.requires_grad_(False)
        text_encoder.to(device, dtype=weight_dtype)

    
    # PREPARE DATASET
    
    print("\n[2/5] Preparing dataset...")

    dataset = DreamBoothDataset(
        subject_dir=config.subject_dir,
        instance_prompt=config.instance_prompt,
        tokenizer=tokenizer,
        class_images_dir=config.class_images_dir if config.prior_preservation else None,
        class_prompt=config.class_prompt if config.prior_preservation else None,
        resolution=config.resolution,
        center_crop=config.center_crop,
    )

    collate_fn = partial(
        dataset.combine,
        prior_preservation=config.prior_preservation,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for small datasets
    )

   
    # 3. SETUP OPTIMIZER
    print("\n[3/5] Setting up optimizer...")

    # collect trainable parameters
    # params_to_optimize = list(unet.parameters())
    # if config.train_text_encoder:
    #     params_to_optimize += list(text_encoder.parameters())

    # optimizer = torch.optim.AdamW(
    #     params_to_optimize,
    #     lr=config.learning_rate,
    #     betas=(config.adam_beta1, config.adam_beta2),
    #     weight_decay=config.adam_weight_decay,
    #     eps=config.adam_epsilon,
    #     foreach=False,
    # )

    params_to_optimize = [
        {
            "params": unet.parameters(),
            "lr": config.learning_rate,  # e.g. 5e-6
        }
    ]

    if config.train_text_encoder:
        params_to_optimize.append(
            {
                "params": text_encoder.parameters(),
                "lr": config.text_encoder_lr,  # e.g. 1e-6
            }
        )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
        foreach=False,
    )

    # learning rate scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.max_train_steps,
    )

    total_params = sum(p.numel() for p in params_to_optimize if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")


    # 4. TRAINING LOOP
  
    print("\n[4/5] Training...")

    global_step = 0
    progress_bar = tqdm(total=config.max_train_steps, desc="Training")
    scaler = torch.cuda.amp.GradScaler(enabled=(weight_dtype == torch.float16))

    while global_step < config.max_train_steps:
        for batch in dataloader:
            if global_step >= config.max_train_steps:
                break

            # move batch to device
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            # encode images to latent space using frozen VAE
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # [DEBUG] check latent health on first step
            if global_step == 0:
                tqdm.write(f"[DEBUG] pixel_values: shape={pixel_values.shape} min={pixel_values.min():.3f} max={pixel_values.max():.3f} nan={pixel_values.isnan().any()}")
                tqdm.write(f"[DEBUG] latents:       shape={latents.shape} mean={latents.float().mean():.3f} std={latents.float().std():.3f} nan={latents.isnan().any()}")

            # sample noise and timesteps
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            # add noise to latents according to the noise schedule
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=(weight_dtype == torch.float16)):
                # get text conditioning
                encoder_hidden_states = text_encoder(input_ids)[0]

                # Unet predicts the noise (denoising)
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # [DEBUG] check noise_pred health on first step
                if global_step == 0:
                    tqdm.write(f"[DEBUG] noise_pred:   shape={noise_pred.shape} mean={noise_pred.float().mean():.3f} std={noise_pred.float().std():.3f} nan={noise_pred.isnan().any()} inf={noise_pred.isinf().any()}")

                # compute total loss
                if config.prior_preservation:
                    # split predictions back into subject and class halves
                    noise_pred_subject, noise_pred_class = torch.chunk(noise_pred, 2, dim=0)
                    noise_subject, noise_class = torch.chunk(noise, 2, dim=0)

                    # compute reconstruction loss on subject images
                    loss_subject = F.mse_loss(
                        noise_pred_subject.float(),
                        noise_subject.float(),
                        reduction="mean",
                    )

                    # compute prior preservation loss on class images
                    loss_prior = F.mse_loss(
                        noise_pred_class.float(),
                        noise_class.float(),
                        reduction="mean",
                    )

                    # combined loss: L = L_subject + λ · L_prior
                    loss = loss_subject + config.prior_loss_weight * loss_prior
                else:
                    # if no prior preservation, compute only reconstruction loss
                    loss = F.mse_loss(
                        noise_pred.float(),
                        noise.float(),
                        reduction="mean",
                    )

            # backprop with gradient scaling to prevent fp16 underflow
            scaler.scale(loss).backward()

            # Gradient clipping for stability
            if config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_optimize, config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            # logging
            if global_step % config.log_every_n_steps == 0:
                loss_val = loss.item()
                log_msg = f"Step {global_step}: loss={loss_val:.4f} scaler_scale={scaler.get_scale():.0f}"
                if config.prior_preservation:
                    log_msg += f" (subject={loss_subject.item():.4f}, prior={loss_prior.item():.4f})"
                if math.isnan(loss_val) or math.isinf(loss_val):
                    log_msg += "  *** WARNING: loss is NaN/Inf — weights are corrupted ***"
                tqdm.write(log_msg)

            # checkpointing 
            if global_step % config.save_every_n_steps == 0:
                save_path = Path(config.output_dir) / f"checkpoint-{global_step}"
                _save_pipeline(config, unet, text_encoder, vae, tokenizer, noise_scheduler, save_path)
                tqdm.write(f"  Saved checkpoint to {save_path}")

    progress_bar.close()

    # [DEBUG] check UNet weights for NaN/Inf before saving
    nan_params, inf_params = 0, 0
    for _, p in unet.named_parameters():
        if p.isnan().any():
            nan_params += 1
        if p.isinf().any():
            inf_params += 1
    print(f"\n[DEBUG] UNet weight health: {nan_params} params with NaN, {inf_params} params with Inf")
    if nan_params > 0 or inf_params > 0:
        print("[DEBUG] *** Weights are corrupted — saved model will produce black images ***")

    # SAVE FINAL MODEL

    print("\n[5/5] Saving final model...")
    final_path = Path(config.output_dir) / "final"
    _save_pipeline(config, unet, text_encoder, vae, tokenizer, noise_scheduler, final_path)
    print(f"  Model saved to {final_path}")
    print("\nTraining complete!")


def _save_pipeline(config, unet, text_encoder, vae, tokenizer, scheduler, save_path):
    """Save the fine-tuned model as a full StableDiffusionPipeline."""
    pipeline = StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DreamBooth")
    parser.add_argument("--subject_dir", type=str, required=True,
                        help="Directory with 3-5 subject images")
    parser.add_argument("--class_dir", type=str, default="./data/class_images",
                        help="Directory with pre-generated class images")
    parser.add_argument("--class_noun", type=str, default="dog",
                        help='Class descriptor (e.g. "dog", "cat", "backpack")')
    parser.add_argument("--identifier", type=str, default="sks",
                        help="Unique identifier token for the subject")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Where to save the fine-tuned model")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--text_encoder_lr", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=1000,
                        help="Max training steps")
    parser.add_argument("--no_prior_preservation", action="store_true",
                        help="Disable prior preservation loss")
    parser.add_argument("--no_train_text_encoder", action="store_true",
                        help="Freeze text encoder during training")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = DreamBoothConfig(
        subject_dir=args.subject_dir,
        class_images_dir=args.class_dir,
        class_noun=args.class_noun,
        identifier_token=args.identifier,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        text_encoder_lr=args.text_encoder_lr,
        max_train_steps=args.steps,
        prior_preservation=not args.no_prior_preservation,
        train_text_encoder=not args.no_train_text_encoder,
        resolution=args.resolution,
        seed=args.seed,
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train(config)