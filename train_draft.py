import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import tifffile
from torch import nn
import argparse 
from stedfm.DEFAULTS import BASE_PATH 
from stedfm import get_pretrained_model_v2 
from torch.utils.data import DataLoader, Dataset
import os 
from tqdm import tqdm, trange 
import glob 
from denoising_unet import UNet 
from diffusion_model_draft import DRaFT_DDPM, RewardEncoder
from segmentation_unet import SegmentationUNet 
from datasets.dendrites_dataset import DendriticFActinDataset 
from datasets.axons_dataset import AxonalRingsDataset
import time 


parser = argparse.ArgumentParser()
### DM general arguments
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets/DendriticFActinDataset"))
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--dry-run", action="store_true")
### DRaFT specific arguments
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--num-sampling-steps", type=int, default=100)
parser.add_argument("--reward-weight", type=float, default=1.0) 
parser.add_argument("--denoising-weight", type=float, default=0.1)
parser.add_argument("--pretrained-checkpoint", type=str, default=os.path.join(BASE_PATH, "baselines/SR-baselines/DM_DendriticFActin/DM_AxonalRings_200.pth"))
parser.add_argument("--use-low-variance", action="store_true")
parser.add_argument("--use-lora", action="store_true", default=True)
parser.add_argument("--no-lora", action="store_false", dest="use_lora")
parser.add_argument("--lora-rank", type=int, default=4)
parser.add_argument("--lora-alpha", type=float, default=1.0)
parser.add_argument("--lora-dropout", type=float, default=0.0)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--use-gradient-checkpointing", action="store_true", default=True, help="Use gradient checkpointing (highly recommended!)")
parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="use_gradient_checkpointing")
parser.add_argument("--use-ema", action="store_true", default=False, help="Use Exponential Moving Average (recommended for stability)")
parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")
args = parser.parse_args()

def validation_step(model: nn.Module, valid_dataset: Dataset, device: torch.device, epoch: int, use_ema: bool = False):
    """
    Validation step: generate samples and compute reward.

    Args:
        use_ema: If True and model has EMA, use EMA weights for validation
    """
    indices = np.random.choice(len(valid_dataset), size=5, replace=False)
    model.eval()

    # Swap to EMA model if available and requested
    if use_ema and hasattr(model, 'use_ema') and model.use_ema:
        model.swap_to_ema()

    os.makedirs("./validation/DRAFT", exist_ok=True)

    rewards = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            confocal, sted, _, _ = valid_dataset[idx]
            confocal = confocal.unsqueeze(0).to(device)
            sted = sted.unsqueeze(0).to(device)

            sample, reward = model.sample_with_reward(
                shape=(1, 1, confocal.shape[2], confocal.shape[3]),
                target_images=sted,
                cond=None,
                segmentation=confocal,
            )

            rewards.append(reward.item())

            sample_np = sample.squeeze().cpu().numpy()
            confocal_np = confocal.squeeze().cpu().numpy()
            sted_np = sted.squeeze().cpu().numpy()
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(confocal_np, cmap="hot", vmin=0, vmax=1)
            axs[0].set_title("Input (Confocal)")
            axs[0].axis("off")

            axs[1].imshow(sample_np, cmap="hot", vmin=0, vmax=1)
            axs[1].set_title(f"Generated (DRaFT)\nReward: {reward.item():.4f}")
            axs[1].axis("off")
            
            axs[2].imshow(sted_np, cmap="hot", vmin=0, vmax=1)
            axs[2].set_title("Target (STED)")
            axs[2].axis("off")

            plt.tight_layout()
            fig.savefig(f"./validation/DRAFT/epoch_{epoch}_sample_{i}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


    avg_reward = np.mean(rewards)
    print(f"\nValidation - Average Reward: {avg_reward:.4f}")

    # Swap back from EMA if we used it
    if use_ema and hasattr(model, 'use_ema') and model.use_ema:
        model.swap_back_from_ema()

    model.train()
    return avg_reward
    


if __name__=="__main__":
    os.makedirs(os.path.join(BASE_PATH, "baselines/DRAFT"), exist_ok=True) 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # unet = SegmentationUNet(in_channels=1, out_channels=2)
    # unet.load_state_dict(torch.load("/home/frederic/TA-GAN/checkpoints/UNet_Axons/axon-pretrained-unet/params.net"))
    # unet.to(DEVICE)
    # unet.eval()

    
    files = glob.glob(os.path.join(args.dataset_path, "train", "*.tif"))
    dataset = DendriticFActinDataset(files=files)
    # dataset = AxonalRingsDataset(files=files)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"[---] Loaded train dataset ({len(dataset)} images) [---]")

    valid_files = glob.glob(os.path.join(args.dataset_path, "valid", "*.tif"))
    valid_dataset = DendriticFActinDataset(files=valid_files)
    # valid_dataset = AxonalRingsDataset(files=valid_files)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False) 
    print(f"[---] Loaded validation dataset ({len(valid_dataset)} images) [---]")

    reward_backbone, cfg = get_pretrained_model_v2(
        name="mae-lightning-small",
        weights="MAE_SMALL_STED",
        blocks="all",
        as_classifier=True,
    )
    reward_model = RewardEncoder(backbone=reward_backbone)
    reward_model.to(DEVICE)
    reward_model.eval()
    print(f"[---] Loaded reward model [---]")

    denoising_model = UNet(
        dim=64, 
        channels=2,
        out_dim=1,
        cond_dim=cfg.dim,
        dim_mults=(1,2,4),
        condition_type=None,
        num_classes=4
    )
    print(f"[---] Loaded denoising U-Net [---]")

    diffusion_model = DRaFT_DDPM(
        denoising_model=denoising_model,
        reward_encoder=reward_model,
        timesteps=1000,
        beta_schedule="linear",
        K=args.K,  # None for full DRaFT, or number of steps for DRaFT-K
        use_low_variance=args.use_low_variance,
        reward_weight=args.reward_weight,
        denoising_weight=args.denoising_weight,
        num_sampling_steps=args.num_sampling_steps,
        eta=0.0,  # Deterministic DDIM
        use_lora=args.use_lora,  # Use LoRA for parameter-efficient training
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=args.use_gradient_checkpointing,  # Critical for memory!
        condition_type=None,
        latent_encoder=reward_backbone,
        concat_segmentation=True,
    )
    # Load pretrained checkpoint if provided
    if args.pretrained_checkpoint is not None:
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=DEVICE)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # When using LoRA, we need to load weights into the base layers
        if args.use_lora:
            print("[---] Loading pretrained weights into LoRA-wrapped model [---]")
            
            # Map pretrained keys to LoRA-wrapped keys
            # Original key: "model.downs.0.0.block1.proj.weight"
            # LoRA key: "model.downs.0.0.block1.proj.conv.weight" or "model.downs.0.0.block1.proj.linear.weight"
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    # Try to map to LoRA structure
                    # Check if this key exists in current model
                    if key in diffusion_model.state_dict():
                        # Key exists as-is (not wrapped by LoRA)
                        new_state_dict[key] = value
                    else:
                        # Try adding .conv or .linear for LoRA-wrapped layers
                        # Check if it's a Conv2d or Linear weight/bias
                        conv_key = key.replace(".weight", ".conv.weight").replace(".bias", ".conv.bias")
                        linear_key = key.replace(".weight", ".linear.weight").replace(".bias", ".linear.bias")
                        
                        if conv_key in diffusion_model.state_dict():
                            new_state_dict[conv_key] = value
                        elif linear_key in diffusion_model.state_dict():
                            new_state_dict[linear_key] = value
                        else:
                            # Keep original key (will be in missing_keys)
                            new_state_dict[key] = value
                else:
                    # Non-model keys (like reward_encoder, etc.)
                    new_state_dict[key] = value
            
            state_dict = new_state_dict
        
        # Load weights (may have missing keys for reward encoder and LoRA, which is fine)
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(state_dict, strict=False)
        
        # Filter out LoRA keys from missing keys (these are expected to be missing)
        non_lora_missing = [k for k in missing_keys if "lora" not in k.lower()]
        
        if non_lora_missing:
            print(f"\tMissing keys (non-LoRA; these should only be from the reward encoder): {non_lora_missing[:10]}...")  # Show first 10
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:10]}...")  # Show first 10
        
        print(f"\tLoaded pretrained weights successfully")
        if args.use_lora:
            print(f"\tLoRA layers initialized")
    
    diffusion_model.to(DEVICE)
    diffusion_model.train()

    # Enable EMA if requested
    if args.use_ema:
        diffusion_model.enable_ema(decay=args.ema_decay)
        print(f"[---] EMA enabled with decay={args.ema_decay} [---]")

    if args.use_lora:
        from lora_layers import get_lora_parameters 
        lora_params = get_lora_parameters(diffusion_model.model)
        optimizer = torch.optim.Adam(lora_params, lr=args.learning_rate, betas=(0.9, 0.99))
        trainable_params = sum(p.numel() for p in lora_params) 
        total_params = sum(p.numel() for p in diffusion_model.model.parameters())
        print(f"[---] Parameter efficiency: [---]")
        print(f"\tTotal parameters: {total_params:,}")
        print(f"\tTrainable (LoRA): {trainable_params:,}")
        print(f"\tReduction: {100 * (1 - trainable_params / total_params):.1f}%\n")
    else:
        optimizer = torch.optim.Adam(diffusion_model.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)

    ################## Training loop ################## 
    print("\n" + "="*60)
    print(f"[---] Training DRaFT with the following configuration [---]")
    print(f"\tK (backprop steps): {'All' if args.K is None else args.K}")
    print(f"\tSampling steps (w/ DDIM): {args.num_sampling_steps}")
    print(f"\tReward weight: {args.reward_weight}")
    print(f"\tDenoising weight: {args.denoising_weight}")
    print(f"\tLearning rate: {args.learning_rate}")
    print(f"\tBatch size: {args.batch_size}")
    print(f"\tUse gradient checkpointing: {args.use_gradient_checkpointing}")
    print(f"\tUse LoRA: {args.use_lora}")
    print(f"\tUse EMA: {args.use_ema}")
    if args.use_lora:
        print(f"\tLoRA rank: {args.lora_rank}")
        print(f"\tLoRA alpha: {args.lora_alpha}")
    print("="*60 + "\n")

    best_reward = -float('inf')

    losses, rewards = [], []
    val_rewards = []
    start_time = time.time()
    for epoch in trange(args.num_epochs, desc="... Training DRaFT ..."):
        diffusion_model.train()
        epoch_losses, epoch_rewards = [], [] 

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")):
            confocal, sted, _, _ = batch 
            confocal = confocal.to(DEVICE)
            sted = sted.to(DEVICE)
            optimizer.zero_grad() 

            results = diffusion_model.draft_training_step(
                target_images=sted, 
                cond=None,
                segmentation=confocal
            )
            loss = results["total_loss"]
            reward = results["reward"]

            loss.backward()

            # Check for NaN gradients
            has_nan = False
            if args.use_lora:
                from lora_layers import get_lora_parameters
                params_to_check = get_lora_parameters(diffusion_model.model)
            else:
                params_to_check = diffusion_model.model.parameters()

            for p in params_to_check:
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break

            if has_nan:
                print(f"Warning: NaN gradients detected at epoch {epoch+1}, batch {batch_idx}. Skipping update.")
                optimizer.zero_grad()
                continue

            # Clip gradients to prevent explosion
            if args.use_lora:
                nn.utils.clip_grad_norm_(get_lora_parameters(diffusion_model.model), max_norm=1.0)
            else:
                nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update EMA if enabled
            if args.use_ema:
                diffusion_model.update_ema()
            
            epoch_losses.append(loss.item())
            epoch_rewards.append(reward.item())

        scheduler.step()

        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean(epoch_rewards)
        losses.append(avg_loss)
        rewards.append(avg_reward)


        print(f"[---] Epoch {epoch+1} Summary [---]")
        print(f"\tAverage Loss: {avg_loss:.4f}")
        print(f"\tAverage Reward: {avg_reward:.4f}")
        print(f"\tLearning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        print("\n[---] Running validation [---]")
        val_reward = validation_step(diffusion_model, valid_dataset, DEVICE, epoch, use_ema=args.use_ema)
        val_rewards.append(val_reward)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(rewards, label="Train")
        ax.plot(val_rewards, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reward")
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"./reward_curves_{args.reward_weight}_{args.denoising_weight}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        checkpoint = {   
            "model": diffusion_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "reward": val_reward,
        }

        if val_reward > best_reward:
            best_reward = val_reward 
            save_path = os.path.join(BASE_PATH, "baselines/DRAFT/DendriticFActin_DRaFT_best.pth")
            torch.save(checkpoint, save_path)
            print(f"[---] Saved best model with reward: {val_reward:.4f} [---]")

            if args.use_lora:
                lora_save_path = os.path.join(BASE_PATH, "baselines/DRAFT/DendriticFActin_DRaFT_best_lora.pth")
                diffusion_model.save_lora_weights(lora_save_path)
                print(f"[---] Saved LoRA weights to {lora_save_path} [---]")

        torch.save(checkpoint, os.path.join(BASE_PATH, f"baselines/DRAFT/DendriticFActin_DRaFT_{epoch+1}.pth"))
        if args.use_lora:
            diffusion_model.save_lora_weights(os.path.join(BASE_PATH, f"baselines/DRAFT/DendriticFActin_DRaFT_{epoch+1}_lora.pth")) 

    print(f"[---] Training completed [---]")
    print(f"[---] Best validation reward: {best_reward:.4f} [---]")
    print(f"[---] Training time: {time.time() - start_time:.2f} seconds [---]")
            

        




