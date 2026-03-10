import numpy as np
import torch 
import argparse 
from stedfm.DEFAULTS import BASE_PATH 
import os 
from torch import nn 
import json
from stedfm import get_pretrained_model_v2 
from denoising_unet import UNet 
from diffusion_model_draft import DRaFT_DDPM, RewardEncoder 
from datasets.axons_dataset import AxonalRingsDataset 
from torch.utils.data import DataLoader 
from typing import List 
from scipy.stats import wilcoxon, ttest_ind, mannwhitneyu
import pickle
from diffusion_model import DDPM 
from tqdm import tqdm, trange 
import glob 
import matplotlib.pyplot as plt 
from metrics import compute_metrics

parser = argparse.ArgumentParser() 
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets", "AxonalRingsDataset"))
parser.add_argument("--ddim-ckpt", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT", "AxonalRings", "DDPM_AxonalRings.pth"))
parser.add_argument("--draft-ckpt", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT", "AxonalRings", "DRAFT_AxonalRings.pth"))
parser.add_argument("--eval-only", action="store_true", default=False)
### DRaFT specific arguments
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--reward-weight", type=float, default=1.0) 
parser.add_argument("--denoising-weight", type=float, default=0.1)
parser.add_argument("--use-low-variance", action="store_true", default=True)
parser.add_argument("--use-lora", action="store_true", default=True)
parser.add_argument("--no-lora", action="store_false", dest="use_lora")
parser.add_argument("--lora-rank", type=int, default=8)
parser.add_argument("--lora-alpha", type=float, default=8)
parser.add_argument("--lora-dropout", type=float, default=0.0)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--use-gradient-checkpointing", action="store_true", default=True, help="Use gradient checkpointing (highly recommended!)")
parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="use_gradient_checkpointing")
parser.add_argument("--n-lv-inner-loops", type=int, default=2, help="Number of DRaFT-LV inner loops (paper uses n=2)")
args = parser.parse_args()

def load_diffusion_models(checkpoint_draft: str, device: torch.device):

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

    ckpt = torch.load(checkpoint_draft, map_location=device, weights_only=False)
    draft_state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
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
    standard_state_dict = {}
    for key, value in draft_state_dict.items():
        if "lora" in key:
            continue  # skip LoRA-specific weights
        remapped = key.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias").replace(".linear.weight", ".weight").replace(".linear.bias", ".bias")
        standard_state_dict[remapped] = value
    model_standard.load_state_dict(standard_state_dict, strict=False)
    model_standard.to(device)
    model_standard.eval()

    # --- DRaFT model ---
    denoising_model_draft = UNet(
        dim=64,
        channels=2,
        out_dim=1,
        cond_dim=cfg.dim,
        dim_mults=(1, 2, 4),
        condition_type=None,
        num_classes=4
    )
    model_draft = DRaFT_DDPM(
        denoising_model=denoising_model_draft,
        reward_encoder=reward_model,
        timesteps=1000,
        beta_schedule="linear",
        K=args.K,
        use_low_variance=args.use_low_variance,
        reward_weight=args.reward_weight,
        denoising_weight=args.denoising_weight,
        num_sampling_steps=100,
        eta=0.0,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=False,
        condition_type=None,
        latent_encoder=reward_backbone,
        concat_segmentation=True,
    )
    model_draft.load_state_dict(draft_state_dict, strict=True)
    model_draft.to(device)
    model_draft.eval()

    return model_standard, model_draft

def inference(
    ddim: nn.Module,
    draft: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
):
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    ddim.to(device)
    draft.to(device)
    ddim.eval()
    draft.eval()

    metrics = {
        "mse": {key: [] for key in ["DDIM", "DRaFT"]},
        "psnr": {key: [] for key in ["DDIM", "DRaFT"]},
        "ssim": {key: [] for key in ["DDIM", "DRaFT"]},
    }
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="... Inference ...")):
            confocal, sted, _, metadata_routing = batch 
            confocal = confocal.to(device)
            sted = sted.to(device)

            sted_np = sted.squeeze().cpu().numpy()
            confocal_np = confocal.squeeze().cpu().numpy() 

            ddim_sample = ddim.ddim_sample_loop(
                shape=(1, 1, 224, 224),
                num_steps=100,
                cond=None,
                segmentation=confocal,
            )
            ddim_sample_np = ddim_sample.squeeze().cpu().numpy() 
            ddim_metrics = compute_metrics(
                truth_image=sted_np,
                prediction_image=ddim_sample_np,
            )
            
            for key in ddim_metrics.keys():
                metrics[key]["DDIM"].append(ddim_metrics[key])
            
            draft_sample, reward = draft.sample_with_reward(
                shape=(1, 1, 224, 224),
                target_images=sted,
                cond=None,
                segmentation=confocal,
            )
            draft_sample_np = draft_sample.squeeze().cpu().numpy()
            draft_metrics = compute_metrics(
                truth_image=sted_np,
                prediction_image=draft_sample_np,
            )
            for key in draft_metrics.keys():
                metrics[key]["DRaFT"].append(draft_metrics[key])

            fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            axs[0].imshow(confocal_np, cmap='hot', vmin=0, vmax=1)
            axs[0].set_title("Confocal")
            axs[1].imshow(sted_np, cmap='hot', vmin=0, vmax=1)
            axs[1].set_title("STED")
            axs[2].imshow(np.clip(ddim_sample_np, 0, 1), cmap='hot', vmin=0, vmax=1)
            axs[2].set_title("DDIM")
            axs[3].imshow(np.clip(draft_sample_np, 0, 1), cmap='hot', vmin=0, vmax=1)
            axs[3].set_title("DRaFT")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "samples",f"sample_{i}.pdf"), dpi=900, bbox_inches='tight')
            plt.close(fig)
    return metrics 

def plot_metrics(metrics: dict, metric_key: str, save_dir: str) -> None:
    colors = {"DDIM": "tab:blue", "DRaFT": "#CC503E"}
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)

    for pos, model_key in enumerate(metrics.keys()):
        data = metrics[model_key]
        x = np.random.normal(loc=pos, scale=0.05, size=len(data))
        ax.scatter(x, data, label=model_key, color=colors[model_key], s=10)
        boxes = ax.boxplot(data, positions=[pos], widths=0.3, showmeans=False, showfliers=False, patch_artist=True, medianprops=dict(color='black', linewidth=1.5),
                           boxprops=dict(facecolor='none'))
        
    ax.legend()
    ax.set_xlabel("Model")
    ax.set_xticks([]) 
    ax.set_ylabel(metric_key)
    plt.savefig(os.path.join(save_dir, f"{metric_key}.pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig) 

def compute_stats(results: dict) -> None:
    for metric_key in results.keys():
        ddim_values = np.array(results[metric_key]["DDIM"])
        draft_values = np.array(results[metric_key]["DRaFT"])
        res = mannwhitneyu(ddim_values, draft_values)
        print(f"{metric_key}: {res.pvalue}")

def main():

    OUTPUT_DIR = os.path.join("./axonalrings-experiment", "results")
    if args.eval_only:
        with open(os.path.join(OUTPUT_DIR, "results.pkl"), "rb") as f:
            results = pickle.load(f)
        compute_stats(results)
        for metric_key in results.keys():
            plot_metrics(results[metric_key], metric_key, OUTPUT_DIR)
    else:

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        files = glob.glob(os.path.join(args.dataset_path, "test", "*.tif"))
        test_dataset = AxonalRingsDataset(files=files)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        model_standard, model_draft = load_diffusion_models(args.draft_ckpt, DEVICE)

        results = inference(
            ddim=model_standard,
            draft=model_draft,
            dataloader=test_dataloader,
            device=DEVICE,
            save_dir=OUTPUT_DIR,
        )

        for metric_key in results.keys():
            plot_metrics(results[metric_key], metric_key, OUTPUT_DIR)

        with open(os.path.join(OUTPUT_DIR, "results.pkl"), "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    
if __name__=="__main__":
    main()