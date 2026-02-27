import numpy as np
import torch
from segmentation_unet import SegmentationUNet
from denoising_unet import UNet 
import matplotlib.pyplot as plt 
import os 
from torch import nn
import tifffile
from diffusion_model import DDPM 
from diffusion_model_draft import DRaFT_DDPM, RewardEncoder
import argparse 
import glob
import random
import time
from tqdm import tqdm 
from torch.utils.data import DataLoader
from stedfm import get_pretrained_model_v2
from tiffwrapper import make_composite
from datasets.dendrites_dataset import DendriticFActinDataset
from metrics import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-standard", type=str, default="/home-local/Frederic/baselines/SR-baselines/DM_AxonalRings_200.pth")
parser.add_argument("--checkpoint-draft", type=str, default="/home-local/Frederic/baselines/SR-baselines/DRAFT/DRaFT_7.pth")
parser.add_argument("--checkpoint-unet", type=str, default="/home-local/Frederic/baselines/pretrained-unet-01/params.net")
parser.add_argument("--dataset-path", type=str, default="/home-local/Frederic/Datasets/DendriticFActinDataset")
parser.add_argument("--split", type=str, default="test")
args = parser.parse_args()

def load_diffusion_models(checkpoint_standard: str, checkpoint_draft: str, device: torch.device):

    reward_backbone, cfg = get_pretrained_model_v2(
        name="mae-lightning-small",
        weights="MAE_SMALL_STED",
        blocks="all",
        as_classifier=True,
        path=None,
        pretrained=False,
        mask_ratio=0.0,
        num_classes=4
    )
    reward_model = RewardEncoder(backbone=reward_backbone)
    reward_model.to(device)
    reward_model.eval()


    denoising_model_standard = UNet(
        dim=64,
        channels=2, 
        out_dim=1,
        cond_dim=cfg.dim,
        dim_mults=(1, 2, 4),
        condition_type=None,
        num_classes=4
    )
    model_standard = DDPM(
        denoising_model=denoising_model_standard,
        timesteps=1000,
        beta_schedule="linear",
        concat_segmentation=True,
        condition_type=None,
        latent_encoder=reward_backbone,
    )
    ckpt = torch.load(checkpoint_standard, map_location=device, weights_only=False)
    model_standard.load_state_dict(ckpt["model"])
    model_standard.to(device)

    denoising_model_draft = UNet(
        dim=64, 
        channels=2,
        out_dim=1,
        cond_dim=cfg.dim,
        dim_mults=(1,2,4),
        condition_type=None,
        num_classes=4
    )

    model_draft = DRaFT_DDPM(
        denoising_model=denoising_model_draft,
        reward_encoder=reward_model,
        timesteps=1000,
        beta_schedule="linear",
        K=5,  # None for full DRaFT, or number of steps for DRaFT-K
        use_low_variance=False,
        reward_weight=1.0,
        denoising_weight=0.1,
        num_sampling_steps=100,
        eta=0.0,  # Deterministic DDIM
        use_lora=True,  # Use LoRA for parameter-efficient training
        lora_rank=4,
        lora_alpha=1.0,
        lora_dropout=0.0,
        use_gradient_checkpointing=True,  # Critical for memory!
        condition_type=None,
        latent_encoder=reward_backbone,
        concat_segmentation=True,
    )
    ckpt = torch.load(checkpoint_draft, map_location=device, weights_only=False)
    model_draft.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model_draft.to(device)
    model_draft.eval()

    return model_standard, model_draft

def load_unet(checkpoint_unet: str, device: torch.device):
    unet = SegmentationUNet(in_channels=1, out_channels=2)
    unet.load_state_dict(torch.load(checkpoint_unet, map_location=device))
    unet.to(device)
    unet.eval()
    return unet

def compare_samples(
    model_standard: nn.Module,
    model_draft: nn.Module,
    unet: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> None:
    metrics = {
        "mse": {key: [] for key in ["DDIM", "DRaFT"]},
        "psnr": {key: [] for key in ["DDIM", "DRaFT"]},
        "ssim": {key: [] for key in ["DDIM", "DRaFT"]},
        "rings_aupr": {key: [] for key in ["DDIM", "DRaFT"]},
        "fibers_aupr": {key: [] for key in ["DDIM", "DRaFT"]},
    }
    for i, batch in enumerate(tqdm(dataloader, desc="... Comparing samples ...")):
        confocal, sted, ground_truth, _ = batch 
        ground_truth = ground_truth.squeeze().cpu().numpy()
        gt_rgb = make_composite(ground_truth, luts=['green', 'magenta'], ranges=[(0,1), (0,1)])
        confocal = confocal.to(device) 
        sted = sted.to(device)

        sted_np = sted.squeeze().cpu().numpy()
        confocal_np = confocal.squeeze().cpu().numpy()

        print("[---] Generating images [---]")
        with torch.no_grad():
            
            avg_metrics = {key: [] for key in metrics.keys()}
            for _ in range(3):
                start_time = time.time()
                sample_standard = model_standard.ddim_sample_loop(
                    shape=(1, 1, 224, 224),
                    num_steps=100,
                    cond=None,
                    segmentation=confocal,
                    progress=False,
                )
                standard_seg = unet(sample_standard).squeeze().cpu().numpy()
                standard_seg_rgb  = make_composite(standard_seg, luts=['green', 'magenta'], ranges=[(0,1), (0,1)])
                print(f"\tDDIM generated in {time.time() - start_time:.2f} seconds")
                sample_standard_np = sample_standard.squeeze().cpu().numpy()

                metrics_ddim = compute_metrics(truth_image=sted_np, prediction_image=sample_standard_np, truth_segmentation=ground_truth, prediction_segmentation=standard_seg)
                for key in metrics_ddim.keys():
                    if metrics_ddim[key] != -1.0:
                        avg_metrics[key].append(metrics_ddim[key])
            for key in metrics_ddim.keys():
                if len(avg_metrics[key]) > 0:
                    metrics[key]["DDIM"].append(np.mean(avg_metrics[key]))

            # start_time = time.time()
            # sample_standard_full = model_standard.p_sample_loop(
            #     shape=(1, 1, 224, 224),
            #     cond=None,
            #     segmentation=confocal,
            #     progress=False,
            # )
            # sample_standard_full_np = sample_standard_full.squeeze().cpu().numpy()
            # standard_seg_full = unet(sample_standard_full).squeeze().cpu().numpy()
            # standard_seg_full_rgb  = make_composite(standard_seg_full, luts=['green', 'magenta'], ranges=[(0,1), (0,1)])
            # print(f"\tDDPM generated in {time.time() - start_time:.2f} seconds")
            avg_metrics = {key: [] for key in metrics.keys()}
            for _ in range(3):
                start_time = time.time()
                sample_draft, reward = model_draft.sample_with_reward(
                    shape=(1, 1, 224, 224),
                    target_images=sted,
                    cond=None,
                    segmentation=confocal,
                )
                draft_seg = unet(sample_draft).squeeze().cpu().numpy()
                draft_seg_rgb  = make_composite(draft_seg, luts=['green', 'magenta'], ranges=[(0,1), (0,1)])
                print(f"\tDRaFT generated in {time.time() - start_time:.2f} seconds")
                sample_draft_np = sample_draft.squeeze().cpu().numpy()


                metrics_draft = compute_metrics(truth_image=sted_np, prediction_image=sample_draft_np, truth_segmentation=ground_truth, prediction_segmentation=draft_seg)
                for key in metrics_draft.keys():
                    if metrics_draft[key] != -1.0:
                        avg_metrics[key].append(metrics_draft[key])
            for key in metrics_draft.keys():
                if len(avg_metrics[key]) > 0:
                    metrics[key]["DRaFT"].append(np.mean(avg_metrics[key]))
        
            # metrics_ddpm = compute_metrics(truth_image=sted_np, prediction_image=sample_standard_full_np, truth_segmentation=ground_truth, prediction_segmentation=standard_seg_full)
            # for key in metrics_ddpm.keys():
            #     if metrics_ddpm[key] != -1.0:
            #         metrics[key]["DDPM"].append(metrics_ddpm[key])
            

            # plot results
            fig, axs = plt.subplots(2, 4, figsize=(15, 10))
            axs[0, 0].imshow(confocal_np, cmap='hot', vmin=0, vmax=1)
            axs[0, 0].set_title("Confocal")
            tifffile.imwrite(f"./{args.split}/raw/sample_{i}_confocal.tif", confocal_np.astype(np.float32))

            axs[0, 1].imshow(sted_np, cmap='hot', vmin=0, vmax=1)
            axs[0, 1].set_title("STED")
            axs[1, 1].imshow(gt_rgb)
            tifffile.imwrite(f"./{args.split}/raw/sample_{i}_sted.tif", sted_np.astype(np.float32))

            axs[0, 2].imshow(sample_standard_np, cmap='hot', vmin=0, vmax=1)
            axs[0, 2].set_title("DDIM")
            axs[1, 2].imshow(standard_seg_rgb) 
            axs[1, 2].set_title(f"{metrics_ddim['rings_aupr']:.2f}, {metrics_ddim['fibers_aupr']:.2f}")
            tifffile.imwrite(f"./{args.split}/raw/sample_{i}_ddim.tif", sample_standard_np.astype(np.float32))

            # axs[0, 3].imshow(sample_standard_full_np, cmap='hot', vmin=0, vmax=1)
            # axs[0, 3].set_title("DDPM")
            # axs[1, 3].imshow(standard_seg_full_rgb)
            # axs[1, 3].set_title(f"{metrics_ddpm['rings_aupr']:.2f}, {metrics_ddpm['fibers_aupr']:.2f}")
            # tifffile.imwrite(f"./{args.split}/raw/sample_{i}_ddpm.tif", sample_standard_full_np.astype(np.float32))

            axs[0, 3].imshow(sample_draft_np, cmap='hot', vmin=0, vmax=1)
            axs[0, 3].set_title("DRaFT")
            axs[1, 3].imshow(draft_seg_rgb)
            axs[1, 3].set_title(f"{metrics_draft['rings_aupr']:.2f}, {metrics_draft['fibers_aupr']:.2f}")
            tifffile.imwrite(f"./{args.split}/raw/sample_{i}_draft.tif", sample_draft_np.astype(np.float32))

            for ax in axs.ravel():
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"./{args.split}/sample_{i}.pdf", dpi=900, bbox_inches='tight')
            plt.close()
            
    return metrics 

def plot_metrics(metrics: dict, metric_key: str) -> None:
    colors = {"DDIM": "tab:blue", "DRaFT": "tab:orange"}
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for pos, model_key in enumerate(metrics.keys()):
        data = metrics[model_key]
        x = np.random.normal(loc=pos, scale=0.05, size=len(data))
        ax.scatter(x, data, label=model_key, color=colors[model_key])
        boxes = ax.boxplot(data, positions=[pos], showmeans=False, patch_artist=True, medianprops=dict(color='black', linewidth=1.5),
                           boxprops=dict(facecolor='none'))
        
    ax.legend()
    ax.set_xlabel("Model")
    ax.set_xticks([]) 
    ax.set_ylabel(metric_key)
    plt.savefig(f"./{args.split}/{metric_key}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig) 

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"./{args.split}", exist_ok=True)
    os.makedirs(f"./{args.split}/raw", exist_ok=True)

    files = glob.glob(os.path.join(args.dataset_path, f"{args.split}", "*.tif"))
    # if args.split == "valid":
    #     files = random.sample(files, min(30, len(files)))
    dataset = DendriticFActinDataset(files=files, split=args.split, coordinates_path="/home-local/Frederic/Datasets/DendriticFActinDataset-exported")
    with open("./index_to_filename.txt", "w") as f:
        for i in range(len(dataset)):
            _, _, _,metadata = dataset[i]
            y, x = metadata['y'], metadata['x']
            f.write(f"{i} ({y}, {x}): {metadata['image_path']}\n")

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # model_standard, model_draft = load_diffusion_models(args.checkpoint_standard, args.checkpoint_draft, DEVICE)

    # unet = load_unet(args.checkpoint_unet, DEVICE)

    # metrics = compare_samples(model_standard, model_draft, unet, dataloader, DEVICE)

    # for metric_key in metrics.keys():
    #     plot_metrics(metrics[metric_key], metric_key)


if __name__=="__main__":
    main()
