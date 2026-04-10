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
import random
from typing import List
from tqdm import tqdm, trange 
import torchvision.transforms as T
import glob 
from pix2pix import Pix2Pix 
from datasets.dendrites_dataset import DendriticFActinDataset 
from datasets.axons_dataset import AxonalRingsDataset 
from utils import AverageMeter, SaveBestPix2PixModel

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="DendriticFActinDataset")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets"))
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--save-folder", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT", "DendriticFActin"))
parser.add_argument("--subsample", type=int, default=None)
args = parser.parse_args()


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def training_logs(
    G_loss_history: List[float],
    D_loss_history: List[float],
    val_loss_history: List[float],
    learning_rate_history: List[float],
    log_dir: str,
):
    os.makedirs(os.path.join(log_dir, "training"), exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_twin = ax.twinx()
    ax.plot(np.arange(0, len(G_loss_history), 1), G_loss_history, color='tab:blue', label="Train GAN loss")
    ax.plot(np.arange(0, len(D_loss_history), 1), D_loss_history, color='tab:red', label="Train D loss")
    ax.plot(np.arange(0, len(val_loss_history), 1), val_loss_history, color='tab:orange', label="Validation")
    ax.set_yscale("log")
    ax_twin.plot(np.arange(0, len(learning_rate_history), 1), learning_rate_history, color='tab:green', label='lr', ls='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax_twin.set_ylabel('Learning Rate')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(log_dir, f"Pix2Pix_training_logs-{args.seed}.png"))
    plt.close()

def validation(
    model: Pix2Pix,
    valid_dataloader: DataLoader,
    epoch: int,
    log_dir: str,
):
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="... Validation ..."):
            model.set_input(batch)
            confocal, sted, ground_truth, metadata = batch  
            confocal = confocal[0].squeeze().cpu().numpy() 
            sted = sted[0].squeeze().cpu().numpy() 
            model.forward()
            pred = model.fake_sted[0].squeeze().cpu().numpy()
            # Keep validation objective aligned with training objective: GAN + reconstruction.
            fake_cat = torch.cat((model.real_conf, model.fake_sted), 1)
            pred_fake = model.netD(fake_cat)
            loss_g_gan = model.criterionGAN(pred_fake, True)
            loss_g_rec = model.criterionL1(model.fake_sted, model.real_sted) * 100.0
            loss = loss_g_gan + loss_g_rec
            val_loss.update(loss.item()) 
    # if (epoch + 1) % 10 == 0:
    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #     axs[0].imshow(confocal, cmap="hot", vmin=0, vmax=1)
    #     axs[1].imshow(sted, cmap="hot", vmin=0, vmax=1)
    #     axs[2].imshow(pred, cmap="hot", vmin=0, vmax=1)
    #     axs[0].set_title("Confocal")
    #     axs[1].set_title("STED")
    #     axs[2].set_title("Predicted")
    #     for ax in axs:
    #         ax.axis("off")
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(log_dir, f"Pix2Pix_validation_{epoch+1}_{args.seed}.png"))
    #     plt.close(fig)
    return val_loss.avg

def train(
    model: Pix2Pix,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    log_dir: str,
    save_dir: str,
):
    model_name = f"Pix2Pix-{args.dataset}-{args.subsample if args.subsample else 'full'}-sample-{args.seed}"
    save_best_model = SaveBestPix2PixModel(save_dir=save_dir, model_name=model_name, maximize=False)
    G_loss_history, D_loss_history = [], [] 
    val_loss_history = []
    learning_rate_history = [] 
    for epoch in trange(num_epochs, desc="... Training epochs ..."): 
        model.train()
        train_G_loss = AverageMeter()
        train_D_loss = AverageMeter()
        for batch in tqdm(train_dataloader, desc="... Training batches ..."):
            model.set_input(batch)
            model.backprop()
            train_G_loss.update(model.loss_G.item())
            train_D_loss.update(model.loss_D.item())
        learning_rate_history.append(model.optimizers[0].param_groups[0]['lr'])
        G_loss_history.append(train_G_loss.avg)
        D_loss_history.append(train_D_loss.avg)
        model.update_learning_rate()

        val_loss = validation(model, valid_dataloader, epoch, log_dir=log_dir)
        val_loss_history.append(val_loss)
        # if (epoch + 1) % 10 == 0:
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'optimizer_state_dict': [optimizer.state_dict() for optimizer in model.optimizers],
        #         'scheduler_state_dict': [scheduler.state_dict() for scheduler in model.schedulers],
        #     }, os.path.join(save_dir, f"Pix2Pix_{args.dataset}-{args.subsample if args.subsample else 'full'}-sample-{args.seed}_epoch_{epoch+1}.pth"))

        if not args.dry_run:
            save_best_model(current_val=val_loss, epoch=epoch, model=model)

        
        training_logs(
            G_loss_history=G_loss_history,
            D_loss_history=D_loss_history,
            val_loss_history=val_loss_history,
            learning_rate_history=learning_rate_history,
            log_dir=log_dir,
        )

def main():
    set_seeds(args.seed)
    LOG_FOLDER = f"./{args.dataset}-experiment/Pix2Pix-{args.subsample if args.subsample else 'full'}-sample"
    os.makedirs(LOG_FOLDER, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DatasetClass = DendriticFActinDataset if args.dataset == "DendriticFActinDataset" else AxonalRingsDataset 

    # Take full dataset or subsample the same way as the DDPM model (to ensure valid comparison)

    files = sorted(glob.glob(os.path.join(args.dataset_path, args.dataset, "train", "*.tif")))
    if args.subsample is not None:
        train_files_path = os.path.join(os.path.dirname(LOG_FOLDER), f"DDPM-{args.subsample}-sample", f"subsampled_files-{args.seed}.txt")
        with open(train_files_path, "r") as f:
            files = [line.strip() for line in f if line.strip()] 

    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ]) 
    train_dataset = DatasetClass(files=files, transform=transform)
    print(f"[---] Training set size: {len(train_dataset)} [---]")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    valid_files_path = os.path.join(os.path.dirname(LOG_FOLDER), "valid_files.txt")
    with open(valid_files_path, "r") as f:
        valid_files = [line.strip() for line in f if line.strip()]

    valid_dataset = DatasetClass(files=valid_files, transform=None)
    print(f"[---] Validation set size: {len(valid_dataset)} [---]")
        
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False) 

    pix2pix_model = Pix2Pix(
        num_epochs=args.num_epochs,
        in_channels=1,
        out_channels=1,
        is_train=True,
    )
    pix2pix_model.to(DEVICE)
    pix2pix_model.train()
    print(f"[---] Loaded Pix2Pix model [---]")

    train(
        model=pix2pix_model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        num_epochs=args.num_epochs,
        device=DEVICE,
        log_dir=LOG_FOLDER, 
        save_dir=args.save_folder
    )


if __name__=="__main__":
    main()
