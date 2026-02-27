import numpy as np 
import tifffile
import torch 
import matplotlib.pyplot as plt
import argparse 
from tiffwrapper import make_composite
from stedfm.DEFAULTS import BASE_PATH
from stedfm import get_pretrained_model_v2
from torch.utils.data import DataLoader
import os
from tqdm import tqdm, trange
import glob
from denoising_unet import UNet
from segmentation_unet import SegmentationUNet
from diffusion_model import DDPM
from datasets.axons_dataset import AxonalRingsDataset
from datasets.dendrites_dataset import DendriticFActinDataset


parser = argparse.ArgumentParser()
# parser.add_argument("--dataset-path", type=str, default="/home/frederic/TA-GAN/AxonalRingsDataset")
parser.add_argument("--dataset-path", type=str, default="/home-local/Frederic/Datasets/AxonalRingsDataset")
parser.add_argument("--dataset", type=str, default="axons")
parser.add_argument("--num-epochs", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

def dm_validation_step(model, valid_dataset: AxonalRingsDataset, device: torch.device, epoch: int):
    os.makedirs(f"./validation/{args.dataset}/epoch_{epoch}", exist_ok=True)
    indices = np.random.choice(len(valid_dataset), size=5, replace=False)
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            confocal, sted, ground_truth, _ = valid_dataset[idx]
            ground_truth = ground_truth.squeeze().cpu().numpy()
            #gt_rgb = make_composite(ground_truth, luts=['green', 'magenta'], ranges=[(0, 1), (0, 1)])
            sted = sted.unsqueeze(0).to(device)
            confocal = confocal.unsqueeze(0).to(device)
            sample = model.p_sample_loop(
                shape=(1, 1, confocal.shape[2], confocal.shape[3]),
                cond=None,
                segmentation=confocal,
                progress=True,
            )
            # seg_pred = unet(sample).squeeze().cpu().numpy()
            # print(seg_pred[0].min(), seg_pred[0].max())
            # print(seg_pred[1].min(), seg_pred[1].max())
            # seg_pred[0] = (seg_pred[0] > 0.4).astype(np.uint8)
            # seg_pred[1] = (seg_pred[1] > 0.25).astype(np.uint8)
            # seg_pred_rgb = make_composite(seg_pred, luts=['green', 'magenta'], ranges=[(0, 1), (0, 1)])
            sample_numpy = sample.squeeze().cpu().numpy()
            sted_numpy = sted.squeeze().cpu().numpy()
            confocal_numpy = confocal.squeeze().cpu().numpy()
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(confocal_numpy, cmap="hot", vmin=0, vmax=1)
            axs[1].imshow(sted_numpy, cmap="hot", vmin=0, vmax=1)
            # axs[1, 1].imshow(gt_rgb, vmin=0, vmax=1)
            axs[2].imshow(sample_numpy, cmap="hot", vmin=0, vmax=1)
            # axs[1, 2].imshow(seg_pred_rgb, vmin=0, vmax=1)
            for ax in axs.ravel():
                ax.axis("off")
            plt.tight_layout()
            fig.savefig(f"./validation/{args.dataset}/epoch_{epoch}/sample_{i}.png", dpi=1200, bbox_inches="tight")
            plt.close(fig)


if __name__=="__main__":
    if args.dataset == "dendrites":
        OUTPUT_DIR = "/home-local/Frederic/baselines/SR-baselines/DM_DendriticFactin"
    else:
        OUTPUT_DIR = "/home-local/Frederic/baselines/SR-baselines/DM_AxonalRings"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # os.makedirs("/home-local/Frederic/baselines/SR-baselines", exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # unet = SegmentationUNet(in_channels=1, out_channels=2)
    # unet.load_state_dict(torch.load("/home/frederic/TA-GAN/checkpoints/UNet_Axons/axon-pretrained-unet/params.net"))
    # unet.to(DEVICE)
    # unet.eval()


    files = glob.glob(os.path.join(args.dataset_path, "train", "*.tif"))
    dataset = AxonalRingsDataset(files=files)
    # dataset = DendriticFActinDataset(files=files)

    print(f"Number of training samples: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    valid_files = glob.glob(os.path.join(args.dataset_path, "valid", "*.tif"))
    valid_dataset = AxonalRingsDataset(files=valid_files)
    # valid_dataset = DendriticFActinDataset(files=valid_files)

    print(f"Number of validation samples: {len(valid_dataset)}")

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
    model_kwargs = {}

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    
    model_name = f"DM_{args.dataset}"
    for epoch in trange(args.num_epochs, desc="... Training ..."):
        diffusion_model.train()
        for batch in tqdm(dataloader, desc="... Batches ..."):
            confocal, sted, ground_truth, _ = batch
           
            ground_truth = ground_truth.to(DEVICE)
            confocal = confocal.to(DEVICE)
            sted = sted.to(DEVICE)
            t = torch.randint(0, 1000, (sted.shape[0],), device=DEVICE).long()

            optimizer.zero_grad()
            losses, model_outputs = diffusion_model(x_0=sted, t=t, cond=None, segmentation=confocal, ground_truth=ground_truth,model_kwargs=model_kwargs)
            loss = losses["loss"].mean()
            loss.backward()
            optimizer.step()

        model_checkpoint =  {
            "model": diffusion_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        if epoch % 10 == 0 and not args.dry_run:
            dm_validation_step(diffusion_model, valid_dataset, DEVICE, epoch)
            torch.save(model_checkpoint, f"{OUTPUT_DIR}/{model_name}_{epoch}.pth")

        torch.save(model_checkpoint, f"{OUTPUT_DIR}/{model_name}.pth")
        scheduler.step()

            