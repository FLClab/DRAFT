import numpy as np 
import torch 
import argparse 
from stedfm.DEFAULTS import BASE_PATH 
import os 
from torch import nn 
from stedfm import get_pretrained_model_v2 
from denoising_unet import UNet 
from diffusion_model_draft import DRaFT_DDPM, RewardEncoder  
from datasets.dendrites_dataset import DendriticFActinDataset 
from torch.utils.data import DataLoader 
from typing import List, Dict, Tuple, Optional 
import pickle 
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind 
from tqdm import tqdm, trange 
from diffusion_model import DDPM 
from tqdm import tqdm, trange 
import glob 
import matplotlib.pyplot as plt
from metrics import compute_metrics 

parser = argparse.ArgumentParser() 
parser.add_argument("--dataset", type=str, default="DendriticFActin")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets"))
parser.add_argument("--subsamples", nargs="+", type=int, default=[50, 100, 300, 500, 1000, 2000, 3000])
parser.add_argument("--eval-only", action="store_true", default=False)
parser.add_argument("--model", type=str, default="DDPM")
parser.add_argument("--ckpt-path", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT"))
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
parser.add_argument("--max-queries", type=int, default=10_000)
args = parser.parse_args()

def load_models(model_type: str, subsample: int):
    base_path = os.path.join(args.ckpt_path, args.dataset)
    model_name = f"DRAFT-{args.dataset}Dataset-{subsample}-sample-1-rank8.pth"
    checkpoint_path = os.path.join(base_path, model_name) 
    ckpt = torch.load(checkpoint_path, weights_only=False)
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
    reward_model.eval()

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
    model_standard.eval()

    if model_type == "DDPM":
        return model_standard 
    else:

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
        model_draft.eval()
        return model_draft



def inference(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    save_dir: str,
    model_type: str,
    subsample: int,
    ):
    model.eval()
    image_outdir = os.path.join(save_dir, f"{model_type}_{subsample}-sample")
    os.makedirs(image_outdir, exist_ok=True)
    results = {key: [] for key in ["mse", "psnr", "ssim"]}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"... Inferring {model_type}_{subsample}-sample samples ..."):
            confocal, sted, _, metadata = batch 
            confocal = confocal.to(device)
            sted = sted.to(device) 
            sted_np = sted.squeeze().cpu().numpy()
            confocal_np = confocal.squeeze().cpu().numpy()

            if model_type == "DDPM":

                sample = model.ddim_sample_loop(
                    shape=(1, 1, 224, 224),
                    num_steps=100,
                    cond=None, 
                    segmentation=confocal,
                    progress=False
                )

            else:
                sample, _ = model.sample_with_reward(
                    shape=(1, 1, 224, 224),                                         
                    target_images=sted,
                    cond=None,
                    segmentation=confocal,
                )
            sample_np = np.clip(sample.squeeze().cpu().numpy(), 0, 1)
            sample_metrics = compute_metrics(truth_image=sted_np, prediction_image=sample_np)
            mse, psnr, ssim = sample_metrics["mse"], sample_metrics["psnr"], sample_metrics["ssim"]
            for key in sample_metrics.keys():
                results[key].append(sample_metrics[key]) 
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(confocal_np, cmap="hot", vmin=0, vmax=1)
            axs[1].imshow(sted_np, cmap="hot", vmin=0, vmax=1)
            axs[2].imshow(sample_np, cmap="hot", vmin=0, vmax=1)
            
            axs[0].set_title("Confocal")
            axs[1].set_title("STED")
            axs[2].set_title(f"MSE: {mse:.4f}\nPSNR: {psnr:.4f}\nSSIM: {ssim:.4f}")         
            for ax in axs:
                
                ax.axis("off")
            plt.tight_layout()
            savename = os.path.basename(metadata["image_path"][0].replace(".tif", ""))
            fig.savefig(os.path.join(image_outdir, f"{savename}.pdf"), dpi=900, bbox_inches="tight")
            plt.close(fig)
         
    return results 

def bootstrap(data: np.ndarray, n_bootstraps: int = 100):
    num_samples = len(data)
    bootstrap_means = [] 
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=num_samples, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    return np.mean(bootstrap_means), np.std(bootstrap_means)

def analyze_results(ddpm_results: dict, draft_results: dict, save_dir: str):
    for metric_key in ["mse", "psnr", "ssim"]:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        ddpm_metric_avgs = [] 
        ddpm_metric_stds = [] 
        draft_metric_avgs = [] 
        draft_metric_stds = []
        for subsample in tqdm(ddpm_results.keys(), desc="... Aggregating results ..."):
            ddpm_data = ddpm_results[subsample][metric_key] 
            ddpm_bootstrap_mean, ddpm_bootstrap_std = bootstrap(ddpm_data)
            ddpm_metric_avgs.append(ddpm_bootstrap_mean)
            ddpm_metric_stds.append(ddpm_bootstrap_std)
            try:
                draft_data = draft_results[subsample][metric_key] 
                draft_bootstrap_mean, draft_bootstrap_std = bootstrap(draft_data)
                draft_metric_avgs.append(draft_bootstrap_mean)
                draft_metric_stds.append(draft_bootstrap_std)
            except:
                continue

        ddpm_metric_avgs = np.array(ddpm_metric_avgs)
        ddpm_metric_stds = np.array(ddpm_metric_stds)
        draft_metric_avgs = np.array(draft_metric_avgs)
        draft_metric_stds = np.array(draft_metric_stds)
        x = np.arange(len(ddpm_results.keys())) 
        x_draft = np.arange(len(draft_results.keys()))
        ax.plot(x, ddpm_metric_avgs, color="tab:blue", marker='o', label="DDPM")
        ax.plot(x_draft, draft_metric_avgs, color="#CC503E", marker='o', label="DRaFT")
        ax.set_xlabel("Subsample size")
        ax.set_ylabel(metric_key)
        ax.set_xticks(x)
        ax.set_xticklabels(list(ddpm_results.keys()))
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{metric_key}.pdf"), dpi=900, bbox_inches="tight")
        plt.close(fig)



def main():
    LOG_FOLDER = os.path.join(f"./{args.dataset}-experiment/results/{args.model}") 
    os.makedirs(LOG_FOLDER, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        LOG_FOLDER = os.path.dirname(LOG_FOLDER)
        ddpm_results = {}
        draft_results = {}
        for subsample in args.subsamples:
            if os.path.exists(os.path.join(LOG_FOLDER, "DDPM", f"DDPM-{subsample}-sample.npz")):
                temp_ddim_results = np.load(os.path.join(LOG_FOLDER, "DDPM", f"DDPM-{subsample}-sample.npz"))
                temp_ddim_results = {key: np.array(temp_ddim_results[key]) for key in temp_ddim_results.keys()}
                ddpm_results[subsample] = temp_ddim_results

            if os.path.exists(os.path.join(LOG_FOLDER, "DRAFT", f"DRAFT-{subsample}-sample.npz")):
                temp_draft_results = np.load(os.path.join(LOG_FOLDER, "DRAFT", f"DRAFT-{subsample}-sample.npz"))
                temp_draft_results = {key: np.array(temp_draft_results[key]) for key in temp_draft_results.keys()}
                draft_results[subsample] = temp_draft_results

        analyze_results(ddpm_results=ddpm_results, draft_results=draft_results, save_dir=LOG_FOLDER)

    else:
        files = glob.glob(os.path.join(args.dataset_path, f"{args.dataset}Dataset", "test", "*.tif"))
        
        test_dataset = DendriticFActinDataset(files=files, split="test", coordinates_path="/home-local/Frederic/Datasets/DendriticFActinDataset-exported")
        print(f"[---] Test dataset size: {len(test_dataset)} [---]")
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        for subsample in args.subsamples:
            
            model = load_models(model_type=args.model, subsample=subsample).to(DEVICE)
            results = inference(model=model, dataloader=test_dataloader, device=DEVICE, save_dir=LOG_FOLDER, model_type=args.model, subsample=subsample)
            np.savez(
                os.path.join(LOG_FOLDER, f"{args.model}-{subsample}-sample.npz"),
                **results
            )
            
                


if __name__ == "__main__":
    main()