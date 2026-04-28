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


def _identifier_token_ids(tokenizer, identifier_token: str) -> list[int]:
    token_ids = tokenizer.encode(identifier_token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Identifier token '{identifier_token}' produced no tokenizer ids")
    return token_ids


def _enable_identifier_embedding_training(tokenizer, text_encoder, identifier_token: str):
    """Train only the embedding rows for the identifier token."""
    token_ids = _identifier_token_ids(tokenizer, identifier_token)
    token_set = sorted(set(token_ids))
    embedding = text_encoder.get_input_embeddings()
    embedding.weight.requires_grad_(True)

    mask = torch.zeros(
        (embedding.weight.shape[0], 1),
        device=embedding.weight.device,
        dtype=embedding.weight.dtype,
    )
    mask[token_set] = 1

    def keep_identifier_rows_only(grad):
        return grad * mask.to(device=grad.device, dtype=grad.dtype)

    embedding.weight.register_hook(keep_identifier_rows_only)
    return [embedding.weight], token_set


def _get_mixed_precision_dtype(config: DreamBoothConfig, device: torch.device):
    if device.type != "cuda":
        return torch.float32
    if config.mixed_precision == "fp16":
        return torch.float16
    if config.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _set_unet_trainable_params(unet, mode: str):
    if mode not in {"full", "attention", "cross_attention"}:
        raise ValueError("--unet_train_mode must be one of: full, attention, cross_attention")

    if mode == "full":
        unet.requires_grad_(True)
        return

    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        if mode == "attention" and (".attn1." in name or ".attn2." in name):
            param.requires_grad_(True)
        elif mode == "cross_attention" and ".attn2." in name:
            param.requires_grad_(True)


def _create_optimizer(config: DreamBoothConfig, optimizer_param_groups):
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            print("  Optimizer: bitsandbytes AdamW8bit")
            return bnb.optim.AdamW8bit(
                optimizer_param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
            )
        except ImportError:
            raise RuntimeError(
                "bitsandbytes is required for the default low-memory optimizer. "
                "Install it with `pip install bitsandbytes`, or pass "
                "`--no_8bit_adam` if you have enough VRAM for torch AdamW."
            )

    print("  Optimizer: torch AdamW")
    return torch.optim.AdamW(
        optimizer_param_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        foreach=False,
    )


def train(config: DreamBoothConfig):
    """Run the full DreamBooth training pipeline."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = _get_mixed_precision_dtype(config, device)
    autocast_enabled = device.type == "cuda" and weight_dtype in {torch.float16, torch.bfloat16}

    print("=" * 60)
    print("DreamBooth Training")
    print("=" * 60)
    print(f"  Instance prompt: '{config.instance_prompt}'")
    print(f"  Class prompt:    '{config.class_prompt}'")
    print(f"  Prior preservation: {config.prior_preservation} (λ={config.prior_loss_weight})")
    print(f"  Learning rate:   {config.learning_rate}")
    print(f"  Max steps:       {config.max_train_steps}")
    print(f"  Train text encoder: {config.train_text_encoder}")
    print(f"  Train identifier embedding: {config.train_identifier_embedding}")
    print(f"  UNet train mode: {config.unet_train_mode}")
    print(f"  8-bit Adam: {config.use_8bit_adam}")
    print(f"  Device: {device}, compute dtype: {weight_dtype}")
    print("=" * 60)


    # LOAD PRETRAINED MODEL COMPONENTS

    print("\n[1/5] Loading pretrained model components...")
    # load model 
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_model_parts(
    config,device, weight_dtype)

    # freeze VAE — fp16 is fine since it is never updated
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    # Keep trainable weights in fp32, but use autocast for activations.
    unet.to(device, dtype=torch.float32)
    unet.train()
    _set_unet_trainable_params(unet, config.unet_train_mode)
    # gradient checkpointing recomputes activations during backward instead of
    # storing them — cuts activation memory ~60% at the cost of ~25% more compute
    unet.enable_gradient_checkpointing()
    if config.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("  xFormers memory-efficient attention: enabled")
        except Exception as exc:
            print(f"  xFormers memory-efficient attention: unavailable ({exc})")

    identifier_embedding_params = []
    identifier_token_ids = []
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)
    if config.train_text_encoder:
        text_encoder.to(device, dtype=torch.float32)
        text_encoder.requires_grad_(True)
        text_encoder.train()
        if hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()
    elif config.train_identifier_embedding:
        # The embedding row is trainable, so keep CLIP in fp32. GradScaler
        # cannot unscale fp16 gradients.
        text_encoder.to(device, dtype=torch.float32)
        text_encoder.train()
        if hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()
        identifier_embedding_params, identifier_token_ids = _enable_identifier_embedding_training(
            tokenizer,
            text_encoder,
            config.identifier_token,
        )
        print(f"  Identifier token ids: {identifier_token_ids}")
        if len(identifier_token_ids) > 1:
            print("  [Note] identifier splits into multiple tokens; a single-token identifier is usually stronger.")

    
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
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    if not unet_params:
        raise RuntimeError(f"No UNet parameters selected for mode '{config.unet_train_mode}'")
    params_to_optimize = list(unet_params)
    optimizer_param_groups = [
        {"params": unet_params, "weight_decay": config.adam_weight_decay},
    ]
    if config.train_text_encoder:
        text_encoder_params = [p for p in text_encoder.parameters() if p.requires_grad]
        params_to_optimize += text_encoder_params
        optimizer_param_groups.append(
            {"params": text_encoder_params, "weight_decay": config.adam_weight_decay}
        )
    elif identifier_embedding_params:
        params_to_optimize += identifier_embedding_params
        # Do not apply AdamW weight decay to the embedding matrix. With masked
        # gradients, decoupled weight decay would still move every token row.
        optimizer_param_groups.append(
            {"params": identifier_embedding_params, "weight_decay": 0.0}
        )

    optimizer = _create_optimizer(config, optimizer_param_groups)

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
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and weight_dtype == torch.float16)
    )

    while global_step < config.max_train_steps:
        for batch in dataloader:
            if global_step >= config.max_train_steps:
                break

            # move batch to device
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            # encode images to latent space using frozen VAE
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=autocast_enabled,
                dtype=weight_dtype,
            ):
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

            with torch.cuda.amp.autocast(
                enabled=autocast_enabled,
                dtype=weight_dtype,
            ):
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
            optimizer.zero_grad(set_to_none=True)

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
                        help="Learning rate (paper: 5e-6 for SD)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Max training steps")
    parser.add_argument("--no_prior_preservation", action="store_true",
                        help="Disable prior preservation loss")
    parser.add_argument("--train_text_encoder", action="store_true",
                        help="Train the full text encoder. High VRAM.")
    parser.add_argument("--no_train_text_encoder", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--no_identifier_embedding", action="store_true",
                        help="Disable low-memory identifier-token embedding training")
    parser.add_argument("--unet_train_mode", type=str, default="cross_attention",
                        choices=["full", "attention", "cross_attention"],
                        help="How much of the UNet to train")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Forward/activation precision")
    parser.add_argument("--no_8bit_adam", action="store_true",
                        help="Disable bitsandbytes AdamW8bit")
    parser.add_argument("--no_xformers", action="store_true",
                        help="Disable xFormers memory-efficient attention")
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
        max_train_steps=args.steps,
        prior_preservation=not args.no_prior_preservation,
        train_text_encoder=args.train_text_encoder and not args.no_train_text_encoder,
        train_identifier_embedding=not args.no_identifier_embedding,
        unet_train_mode=args.unet_train_mode,
        mixed_precision=args.mixed_precision,
        use_8bit_adam=not args.no_8bit_adam,
        enable_xformers=not args.no_xformers,
        resolution=args.resolution,
        seed=args.seed,
    )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train(config)
