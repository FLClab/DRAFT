import numpy as np 
import torch 
import argparse
from stedfm.DEFAULTS import BASE_PATH 
import os 
from torch import nn
from stedfm import get_pretrained_model_v2
from denoising_unet import UNet
from diffusion_model import DDPM
from diffusion_model_draft import DRaFT_DDPM, RewardEncoder
from datasets.axons_dataset import AxonalRingsDataset
from torch.utils.data import DataLoader
import glob
from utils import AverageMeter, SaveBestModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets", "AxonalRingsDataset"))
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--save-folder", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT", "AxonalRings"))
parser.add_argument("--ddim-ckpt", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT", "AxonalRings", "DDPM_AxonalRings.pth"))
### DRaFT specific arguments
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--num-sampling-steps", type=int, default=100)
parser.add_argument("--reward-weight", type=float, default=1.0) 
parser.add_argument("--denoising-weight", type=float, default=0.1)
parser.add_argument("--use-low-variance", action="store_true", default=True)
parser.add_argument("--use-lora", action="store_true", default=True)
parser.add_argument("--no-lora", action="store_false", dest="use_lora")
parser.add_argument("--lora-rank", type=int, default=4)
parser.add_argument("--lora-alpha", type=float, default=1.0)
parser.add_argument("--lora-dropout", type=float, default=0.0)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--use-gradient-checkpointing", action="store_true", default=True, help="Use gradient checkpointing (highly recommended!)")
parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="use_gradient_checkpointing")
args = parser.parse_args()


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int,
    device: torch.device,
    log_dir: str,
    save_dir: str,
):
    pass

def main():
    LOG_FOLDER = "./axonalrings-experiment/DRAFT"
    os.makedirs(LOG_FOLDER, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = glob.glob(os.path.join(args.dataset_path, "train", "*.tif"))
    train_dataset = AxonalRingsDataset(files=files)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    valid_files = glob.glob(os.path.join(args.dataset_path, "valid", "*.tif"))
    valid_dataset = AxonalRingsDataset(files=valid_files)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False) 

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
    if args.ddim_ckpt is not None:
        checkpoint = torch.load(args.ddim_ckpt, map_location=DEVICE)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
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
    if args.use_lora:
        print(f"\tLoRA rank: {args.lora_rank}")
        print(f"\tLoRA alpha: {args.lora_alpha}")
    print("="*60 + "\n")

    train(
        model=diffusion_model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer, 
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=DEVICE,
        log_dir=LOG_FOLDER,
        save_dir=args.save_folder,
    )
    


if __name__ == "__main__":
    main()
