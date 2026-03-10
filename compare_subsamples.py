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

def load_ddpm(model_type: str, subsample: int):
    base_path = os.path.join(args.ckpt_path, args.dataset)
    model_name = f"{model_type}_{args.dataset}Dataset-{subsample}-sample.pth"
    checkpoint_path = os.path.join(base_path, model_name) 
    checkpoint = torch.load(checkpoint_path, weights_only=False)
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
    diffusion_model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(f"[---] Loaded best model for {subsample} from epoch {checkpoint['epoch']} [---]")
    diffusion_model.eval()
    return diffusion_model 

def load_draft(model_type: str, subsample: int):
    pass 

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

            sample = model.ddim_sample_loop(
                shape=(1, 1, 224, 224),
                num_steps=100,
                cond=None, 
                segmentation=confocal,
                progress=False
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

def analyze_results(ddpm_results: dict, save_dir: str):
    for metric_key in ["mse", "psnr", "ssim"]:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        metric_avgs = [] 
        metric_stds = [] 
        for subsample in tqdm(ddpm_results.keys(), desc="... Aggregating results ..."):
            data = ddpm_results[subsample][metric_key] 
            bootstrap_mean, bootstrap_std = bootstrap(data)
            metric_avgs.append(bootstrap_mean)
            metric_stds.append(bootstrap_std)
        metric_avgs = np.array(metric_avgs)
        metric_stds = np.array(metric_stds)
        x = np.arange(len(ddpm_results.keys())) 
        ax.plot(x, metric_avgs, color="tab:blue", marker='o', label="DDPM")
        ax.fill_between(x, metric_avgs - metric_stds, metric_avgs + metric_stds, color="tab:blue", alpha=0.2)
        ax.set_xlabel("Subsample size")
        ax.set_ylabel(metric_key)
        ax.set_xticks(x)
        ax.set_xticklabels(list(ddpm_results.keys()))
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{metric_key}.pdf"), dpi=900, bbox_inches="tight")
        plt.close(fig)



def main():
    LOG_FOLDER = os.path.join(f"./{args.dataset}-experiment/results") 
    os.makedirs(LOG_FOLDER, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        ddpm_results = {}
        for subsample in args.subsamples:
            temp_results = np.load(os.path.join(LOG_FOLDER, f"{args.model}-{subsample}-sample.npz"))
            temp_results = {key: np.array(temp_results[key]) for key in temp_results.keys()}
            ddpm_results[subsample] = temp_results
        analyze_results(ddpm_results=ddpm_results, save_dir=LOG_FOLDER)

    else:
        files = glob.glob(os.path.join(args.dataset_path, f"{args.dataset}Dataset", "test", "*.tif"))
        
        test_dataset = DendriticFActinDataset(files=files, split="test", coordinates_path="/home-local/Frederic/Datasets/DendriticFActinDataset-exported")
        print(f"[---] Test dataset size: {len(test_dataset)} [---]")
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        for subsample in args.subsamples:
            model = load_ddpm(model_type=args.model, subsample=subsample).to(DEVICE)
            results = inference(model=model, dataloader=test_dataloader, device=DEVICE, save_dir=LOG_FOLDER, model_type=args.model, subsample=subsample)
            np.savez(
                os.path.join(LOG_FOLDER, f"{args.model}-{subsample}-sample.npz"),
                **results
            )


if __name__ == "__main__":
    main()