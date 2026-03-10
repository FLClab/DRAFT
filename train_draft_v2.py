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
from typing import List
from tqdm import tqdm, trange
import torchvision.transforms as T
import glob
import random
from datasets.dendrites_dataset import DendriticFActinDataset
import matplotlib.pyplot as plt
from utils import AverageMeter, SaveBestModel

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="DendriticFActinDataset")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets", "DendriticFActinDataset"))
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--save-folder", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT", "DendriticFActin"))
parser.add_argument("--ddim-ckpt", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT", "DendriticFActin", "DDPM_AxonalRings.pth"))
### DRaFT specific arguments
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--num-sampling-steps", type=int, default=100)
parser.add_argument("--reward-weight", type=float, default=1.0) 
parser.add_argument("--denoising-weight", type=float, default=0.1)
parser.add_argument("--use-low-variance", action="store_true", default=False)
parser.add_argument("--use-lora", action="store_true", default=True)
parser.add_argument("--no-lora", action="store_false", dest="use_lora")
parser.add_argument("--lora-rank", type=int, default=8)
parser.add_argument("--lora-alpha", type=float, default=8)
parser.add_argument("--lora-dropout", type=float, default=0.0)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--use-gradient-checkpointing", action="store_true", default=True, help="Use gradient checkpointing (highly recommended!)")
parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="use_gradient_checkpointing")
parser.add_argument("--n-lv-inner-loops", type=int, default=2, help="Number of DRaFT-LV inner loops (paper uses n=2)")
parser.add_argument("--subsample", type=int, default=None)
args = parser.parse_args()


def training_queries(
    queries: List[int],
    rewards: List[float],
    log_dir: str,
):
    os.makedirs(os.path.join(log_dir, "training"), exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(queries), rewards, color='firebrick', label="Rewards")
    ax.set_xlabel('Queries')
    ax.set_ylabel('Rewards / MSE')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(log_dir, "training", f"DRAFT-{args.K}_training_queries_{args.subsample if args.subsample else 'full'}-sample.png"))
    plt.close()
    


def training_logs(
    train_loss_history: List[float],
    val_loss_history: List[float],
    learning_rate_history: List[float],
    log_dir: str,
):
    os.makedirs(os.path.join(log_dir, "training"), exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_twin = ax.twinx()
    ax.plot(np.arange(0, len(train_loss_history), 1), train_loss_history, color='tab:blue', label="Train")
    ax.plot(np.arange(0, len(val_loss_history), 1), val_loss_history, color='tab:orange', label="Validation")
    ax_twin.plot(np.arange(0, len(learning_rate_history), 1), learning_rate_history, color='tab:green', label='lr', ls='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax_twin.set_ylabel('Learning Rate')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(log_dir, "training", f"DRAFT-{args.K}_training_logs_{args.subsample if args.subsample else 'full'}-sample.png"))
    plt.close()


def display_samples(
    model: nn.Module,
    confocals: torch.Tensor,
    steds: torch.Tensor,
    epoch: int,
    log_dir: str,
    device: torch.device,
):
    os.makedirs(f"{log_dir}/{args.subsample if args.subsample else 'full'}-sample/epoch_{epoch+1}", exist_ok=True)
    for i in range(confocals.shape[0]):
        conf = confocals[i].unsqueeze(0).to(device)
        with torch.no_grad():
            sample, reward = model.sample_with_reward(
                shape=(1, 1, conf.shape[2], conf.shape[3]),
                target_images=steds[i].unsqueeze(0).to(device),
                cond=None,
                segmentation=conf,
            )
            sample_np = sample.squeeze().cpu().numpy() 
            confocal_np = conf.squeeze().cpu().numpy()
            sted_np = steds[i].squeeze().cpu().numpy()
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(confocal_np, cmap="hot", vmin=0, vmax=1)
            axs[0].set_title("Input (Confocal)")
            axs[0].axis("off")
            axs[1].imshow(sample_np, cmap="hot", vmin=0, vmax=1)
            axs[1].set_title(f"DRaFT\nReward: {reward.item():.4f}")
            axs[1].axis("off")
            axs[2].imshow(sted_np, cmap="hot", vmin=0, vmax=1)
            axs[2].set_title("Target (STED)")
            axs[2].axis("off")
            plt.tight_layout()
            fig.savefig(f"{log_dir}/{args.subsample if args.subsample else 'full'}-sample/epoch_{epoch+1}/sample_{i}.png", dpi=1200, bbox_inches="tight")
            plt.close(fig)
            if i == 10:
                break 
    



def validation(
    model: nn.Module,
    dataloader: DataLoader, 
    device: torch.device,
    epoch: int,
    log_dir: str, 
    display: bool = True
):
    model.eval() 
    loss_meter = AverageMeter()
    reward_meter = AverageMeter()
    mse_meter = AverageMeter()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="... Validation batches ..."):
            confocal, sted, _, metadata = batch 
            confocal = confocal.to(device)
            sted = sted.to(device)

            results = model.draft_training_step(
                target_images=sted, 
                cond=None, 
                segmentation=confocal,
            )

            loss = results["total_loss"]
            mse_loss = results["denoising_loss"]
            reward = results["reward"]

            loss_meter.update(loss.item())
            mse_meter.update(mse_loss.item())
            reward_meter.update(reward.item())

    if display:
        display_samples(model, confocal, sted, epoch, log_dir, device)

    return loss_meter.avg, mse_meter.avg, reward_meter.avg


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
    model_name = f"DRAFT-{args.dataset}-{args.subsample if args.subsample else 'full'}-sample-{args.K}-rank{args.lora_rank}"
    if args.use_low_variance and args.K == 1:
        model_name += "_LV"
    save_best_model = SaveBestModel(save_dir=save_dir, model_name=model_name, maximize=False)
    loss_history, val_loss_history = [], []
    mse_loss_history, val_mse_loss_history = [], []
    reward_history, val_reward_history = [], []
    learning_rate_history = []
    num_queries = 0
    for epoch in trange(num_epochs, desc="... Training epochs ..."):
        model.train() 
        train_loss = AverageMeter()
        for batch in tqdm(train_dataloader, desc="... Training batches ..."):
            confocal, sted, _, metadata = batch
            confocal = confocal.to(device)
            sted = sted.to(device)
            optimizer.zero_grad()

            results = model.draft_training_step(
                target_images=sted,
                cond=None,
                segmentation=confocal,
            )
            loss = results["total_loss"]
            mse_loss = results["denoising_loss"]
            reward = results["reward"]

            loss.backward()

            has_nan = False
            if args.use_lora:
                from lora_layers import get_lora_parameters
                params_to_check = get_lora_parameters(model.model)
            else:
                params_to_check = model.model.parameters()

            for p in params_to_check:
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break

            if has_nan:
                print(f"Warning: NaN gradients detected at epoch {epoch+1}. Skipping update.")
                optimizer.zero_grad()
                continue

            # Clip gradients to prevent explosion
            if args.use_lora:
                nn.utils.clip_grad_norm_(get_lora_parameters(model.model), max_norm=1.0)
            else:
                nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            


            optimizer.step()
            train_loss.update(loss.item())
            mse_loss_history.append(mse_loss.item())
            reward_history.append(reward.item()) 
            num_queries += 1

       
        current_learning_rate = optimizer.param_groups[0]["lr"]
        learning_rate_history.append(current_learning_rate)
        scheduler.step()
        loss_history.append(train_loss.avg)
        
        val_loss, val_mse_loss, val_reward = validation(model, valid_dataloader, device, epoch, log_dir=log_dir, display=(epoch % 10 == 0))
        val_loss_history.append(val_loss)
        val_mse_loss_history.append(val_mse_loss)
        val_reward_history.append(val_reward)

        training_logs(
            train_loss_history=loss_history,
            val_loss_history=val_loss_history,
            learning_rate_history=learning_rate_history,
            log_dir=log_dir,
        )
        training_queries(
            queries=num_queries,
            rewards=reward_history,
            log_dir=log_dir,
        )

        if (epoch + 1)% 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(save_dir, f"DRAFT-{args.dataset}-{args.subsample if args.subsample else 'full'}-sample-{args.K}-rank{args.lora_rank}_epoch_{epoch+1}.pth"))

        if not args.dry_run:
            save_best_model(
                current_val=val_loss,
                epoch=epoch,
                model=model, 
                optimizer=optimizer,
                criterion=val_loss,
            )


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    set_seeds(args.seed)
    LOG_FOLDER = f"./{args.dataset}-experiment/DRAFT-{args.K}-rank{args.lora_rank}-{args.subsample if args.subsample else 'full'}-sample"
    os.makedirs(LOG_FOLDER, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DatasetClass = DendriticFActinDataset if args.dataset == "DendriticFActinDataset" else AxonalRingsDataset
    
    files = glob.glob(os.path.join(args.dataset_path, "train", "*.tif"))
    if args.subsample is not None:
        files = random.sample(files, args.subsample)

    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ])
    train_dataset = DatasetClass(files=files, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    valid_files = glob.glob(os.path.join(args.dataset_path, "valid", "*.tif"))
    valid_files = random.sample(valid_files, 100)
    valid_dataset = DatasetClass(files=valid_files, transform=None)
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
        n_lv_inner_loops=args.n_lv_inner_loops,
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
        optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.1)
        trainable_params = sum(p.numel() for p in lora_params) 
        total_params = sum(p.numel() for p in diffusion_model.model.parameters())
        print(f"[---] Parameter efficiency: [---]")
        print(f"\tTotal parameters: {total_params:,}")
        print(f"\tTrainable (LoRA): {trainable_params:,}")
        print(f"\tReduction: {100 * (1 - trainable_params / total_params):.1f}%\n")
    else:
        optimizer = torch.optim.AdamW(diffusion_model.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.1)
    
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
    if args.use_low_variance and args.K == 1:
        print(f"\tDRaFT-LV inner loops: {args.n_lv_inner_loops}")
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
