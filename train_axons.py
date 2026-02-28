import numpy as np 
import tifffile 
import torch 
from torch import nn
import matplotlib.pyplot as plt 
import argparse 
from tiffwrapper import make_composite 
from stedfm.DEFAULTS import BASE_PATH 
from stedfm import get_pretrained_model_v2 
from torch.utils.data import DataLoader 
import os 
from typing import List
from tqdm import tqdm, trange 
import torchvision.transforms as T
import glob 
from denoising_unet import UNet
from diffusion_model import DDPM 
from datasets.axons_dataset import AxonalRingsDataset 
from utils import AverageMeter, SaveBestModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets", "AxonalRingsDataset"))
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--save-folder", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT", "AxonalRings"))
args = parser.parse_args()



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
    fig.savefig(os.path.join(log_dir, "DDPM_training_logs.png"))
    plt.close()

def display_samples(
    model: nn.Module,
    confocals: torch.Tensor,
    steds: torch.Tensor,
    epoch: int,
    log_dir: str,
):
    os.makedirs(f"{log_dir}/epoch_{epoch+1}", exist_ok=True)
    for i in range(confocals.shape[0]):
        conf = confocals[i].unsqueeze(0)
        with torch.no_grad():
            sample = model.ddim_sample_loop(
                shape=(1, 1, conf.shape[2], conf.shape[3]),
                cond=None,
                num_steps=100,
                segmentation=conf,
                progress=True,
            )
        sample_numpy = np.clip(sample.squeeze().cpu().numpy(), 0, 1)

        conf_numpy = conf.squeeze().cpu().numpy()
        sted_numpy = steds[i].squeeze().cpu().numpy()
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(conf_numpy, cmap="hot", vmin=0, vmax=1)
        axs[1].imshow(sted_numpy, cmap="hot", vmin=0, vmax=1)
        axs[2].imshow(sample_numpy, cmap="hot", vmin=0, vmax=1)
        for ax in axs.ravel():
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(f"{log_dir}/epoch_{epoch+1}/sample_{i}.png", dpi=1200, bbox_inches="tight")
        plt.close(fig)
        if i == 10:
            break
    

def validation(
    model: nn.Module,
    valid_dataloader: DataLoader, 
    device: torch.device,
    epoch: int,
    log_dir: str,
    display: bool = False,
):
    model.eval()
    loss_meter = AverageMeter()
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="... Validation ..."):
            confocal, sted, _, metadata = batch
            confocal = confocal.to(device)
            sted = sted.to(device)
            t = torch.randint(0, 1000, (sted.shape[0],), device=device).long()
            losses, model_outputs = model(
                x_0=sted, 
                t=t, 
                cond=None,
                segmentation=confocal,
                model_kwargs={},
            )
            loss = losses["loss"].mean()
            loss_meter.update(loss.item())
    if display:
        display_samples(model=model, confocals=confocal, steds=sted,epoch=epoch, log_dir=log_dir)
    return loss_meter.avg

def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    log_dir: str,
    save_dir: str,
):
    model_kwargs = {}
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    model_name = "DDPM_AxonalRings"

    save_best_model = SaveBestModel(save_dir=save_dir, model_name=model_name, maximize=False) 
    loss_history, val_loss_history = [], [] 
    learning_rate_history = [] 
    for epoch in trange(num_epochs, desc="... Training epochs ..."):
        model.train()
        train_loss = AverageMeter()
        for batch in tqdm(train_dataloader, desc="... Training batches ..."):
            confocal, sted, _, metadata = batch 
            confocal = confocal.to(device)
            sted = sted.to(device)

            t = torch.randint(0, 1000, (sted.shape[0],), device=device).long()
            optimizer.zero_grad()
            losses, model_outputs = model(
                x_0=sted, 
                t=t, 
                cond=None,
                segmentation=confocal,
                model_kwargs=model_kwargs,
            )
            loss = losses["loss"].mean()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
        learning_rate_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        loss_history.append(train_loss.avg)
        
        val_loss = validation(model, valid_dataloader, device, epoch, log_dir=log_dir, display=(epoch % 10 == 0))
        val_loss_history.append(val_loss)

        if (epoch + 1) % 10 == 0:
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, os.path.join(save_dir, f"DDPM_AxonalRings_epoch_{epoch+1}.pth"))

        if not args.dry_run:
            save_best_model(current_val=val_loss, epoch=epoch, model=model, optimizer=optimizer, criterion=val_loss)

        training_logs(
            train_loss_history=loss_history,
            val_loss_history=val_loss_history,
            learning_rate_history=learning_rate_history,
            log_dir=log_dir,
        )

def main():
    os.makedirs(args.save_folder, exist_ok=True)
    LOG_FOLDER = "./axonalrings-experiment"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    files = glob.glob(os.path.join(args.dataset_path, "train", "*.tif"))
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ])
    train_dataset = AxonalRingsDataset(files=files, transform=transform)
    print(f"[---] Training set size: {len(train_dataset)} [---]")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False) 

    valid_files = glob.glob(os.path.join(args.dataset_path, "valid", "*.tif")) 
    valid_dataset = AxonalRingsDataset(files=valid_files, transform=None)
    print(f"[---] Validation set size: {len(valid_dataset)} [---]")
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size*8, shuffle=False, drop_last=False) 

    latent_encoder, cfg = get_pretrained_model_v2(
        name="mae-lightning-small",
        weights="MAE_SMALL_STED",
        blocks="all",
        path=None,
        mask_ratio=0.0,
        pretrained=False,
        as_classifier=True,
        num_classes=4
    )
    latent_encoder.eval()
    latent_encoder.to(DEVICE) 

    denoising_model = UNet(
        dim=64, 
        channels=2,
        out_dim=1,
        cond_dim=cfg.dim,
        dim_mults=(1,2,4),
        condition_type=None,
        num_classes=4
    )
    
    diffusion_model = DDPM(
        denoising_model=denoising_model, 
        timesteps=1000,
        beta_schedule="linear",
        condition_type=None,
        latent_encoder=latent_encoder,
        concat_segmentation=True,
    ) 

    diffusion_model.to(DEVICE)
    diffusion_model.train()
    


    train(
        model=diffusion_model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        num_epochs=args.num_epochs,
        device=DEVICE, 
        log_dir=LOG_FOLDER,
        save_dir=args.save_folder,
    )


if __name__=="__main__":
    main()