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
from QualityNet.networks import NetTrueFCN 
from typing import List, Dict, Tuple, Optional, Union
import pickle 
from banditopt.objectives import Resolution 
from collections import defaultdict
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind 
from tqdm import tqdm, trange 
from diffusion_model import DDPM 
from tqdm import tqdm, trange 
from skimage.filters import threshold_otsu
import tifffile
from tiffwrapper import make_composite
import glob 
from pix2pix import Pix2Pix
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from metrics import compute_metrics 
from segmentation_unet import SegmentationUNet

parser = argparse.ArgumentParser() 
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="DendriticFActin")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets"))
parser.add_argument("--subsamples", nargs="+", type=int, default=[50, 100, 250, 300, 500, 1000, 2000, 3000, None])
parser.add_argument("--eval-only", action="store_true", default=False)
parser.add_argument("--model", type=str, default="DDPM")
parser.add_argument("--ckpt-path", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT"))
parser.add_argument("--unet-checkpoint", type=str, default=os.path.join(BASE_PATH, "baselines", "pretrained-unet-01", "params.net"))
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
args = parser.parse_args()

METRIC_NAMES = ["mse", "mae", "psnr", "ssim", "ms_ssim", "rings_dice", "fibers_dice", "quality", "resolution", "fourier_ncc", "wavelet_ncc", "phase_correlation", "stedfm"]

def load_quality_net() -> nn.Module:
    quality_net = NetTrueFCN()
    quality_checkpoint = torch.load(f"./QualityNet/trained_models/actin/qualitynet.pth", weights_only=False)
    quality_net.load_state_dict(quality_checkpoint["model_state_dict"])
    return quality_net

def infer_quality(img: np.ndarray, quality_net: nn.Module) -> float:
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    quality_net.eval()
    with torch.no_grad():
        score = quality_net(img)
    return score.item()

def compute_resolution(img: np.ndarray) -> float:
    if img.shape[0] == 3:
        img = img[0]
    resolution_objective = Resolution(pixelsize=20e-9)
    resolution = resolution_objective.evaluate([img], None, None, None, None)
    return resolution

def load_ddpm(subsample: int):
    base_path = os.path.join(args.ckpt_path, args.dataset)
    model_name = f"DDPM_{args.dataset}Dataset-{subsample}-sample-{args.seed}.pth"
    checkpoint_path = os.path.join(base_path, model_name)
    ckpt = torch.load(checkpoint_path, weights_only=False)
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
        timesteps=100,
        beta_schedule="linear",
        condition_type=None,
        latent_encoder=latent_encoder,
        concat_segmentation=True,
    )
    diffusion_model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"[---] Loaded best DDPM model for {subsample} from epoch {ckpt['epoch']} [---]")
    diffusion_model.eval()
    return diffusion_model
    
def load_pix2pix(subsample: Union[int, str]):
    base_path = os.path.join(args.ckpt_path, args.dataset)
    model_name = f"Pix2Pix-{args.dataset}Dataset-{subsample}-sample-{args.seed}.pth"
    checkpoint_path = os.path.join(base_path, model_name)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = Pix2Pix(num_epochs=100, in_channels=1, out_channels=1)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model

    

def load_models(model_type: str, subsample: int):
    if model_type == "Pix2Pix":
        return load_pix2pix(subsample=subsample)

    base_path = os.path.join(args.ckpt_path, args.dataset)
    model_name = f"DRAFT-{args.dataset}Dataset-{subsample}-sample-{args.seed}-rank8.pth"
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

def compute_segmentation(sample: torch.Tensor, unet: nn.Module, device: torch.device):
    raw = unet(sample).squeeze().cpu().numpy()
    segmentation = np.zeros_like(raw)
    segmentation[0] = raw[0] > 0.25 
    segmentation[1] = raw[1] > 0.4
    return segmentation.astype(np.uint8)

def inference(
    model: nn.Module, 
    dataloader: DataLoader, 
    unet: nn.Module,
    device: torch.device, 
    save_dir: str,
    image_dir: str,
    model_type: str,
    subsample: int,
    quality_net: nn.Module,
    ):
    model.eval()
    image_outdir = os.path.join(save_dir, f"{model_type}_{subsample}-sample-{args.seed}")
    if args.seed in [42, 97]:
        os.makedirs(image_outdir, exist_ok=True)
    results = {key: [] for key in METRIC_NAMES + ["filenames", "confoncals", "steds", "samples"]}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"... Inferring {model_type}_{subsample}-sample samples ..."):
            confocal, sted, ground_truth, metadata = batch 
            ground_truth = ground_truth.squeeze().cpu().numpy() 
            gt_rgb = make_composite(ground_truth, luts=['green', 'magenta'], ranges=[(0,1), (0,1)])
            confocal = confocal.to(device)
            sted = sted.to(device) 
            # tiff_data = []
            sted_np = sted.squeeze().cpu().numpy()
            confocal_np = confocal.squeeze().cpu().numpy()
            results["filenames"].append(metadata["image_path"][0])
            results["confoncals"].append(confocal_np)
            results["steds"].append(sted_np)
            foreground = confocal_np > threshold_otsu(confocal_np)
            # tiff_data.append(confocal_np)
            # tiff_data.append(sted_np)

            if model_type == "Pix2Pix":
                model.set_input(batch)
                model.forward()
                sample = model.fake_sted

            elif model_type == "DDPM":

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

            segmentation = compute_segmentation(sample=sample, unet=unet, device=device)
            segmentation_rgb = make_composite(segmentation, luts=['green', 'magenta'], ranges=[(0,1), (0,1)])
            
            sample_np = np.clip(sample.squeeze().cpu().numpy(), 0, 1)
            results["samples"].append(sample_np)
 
            residual_image = sted_np - sample_np
            quality = infer_quality(img=sample_np, quality_net=quality_net)
            resolution = compute_resolution(img=sample_np)
            results["quality"].append(quality)
            results["resolution"].append(resolution)

            # tiff_data.append(sample_np)
            sample_metrics = compute_metrics(
                truth_image=sted_np, 
                prediction_image=sample_np, 
                foreground=None,
                truth_segmentation=ground_truth,
                prediction_segmentation=segmentation,
                )
            mse, mae, psnr, ssim, ms_ssim, rings_dice, fibers_dice, fourier_ncc, wavelet_ncc, phase_correlation, stedfm_sim = sample_metrics["mse"], sample_metrics["mae"], sample_metrics["psnr"], sample_metrics["ssim"], sample_metrics["ms_ssim"], sample_metrics["rings_dice"], sample_metrics["fibers_dice"], sample_metrics["fourier_ncc"], sample_metrics["wavelet_ncc"], sample_metrics["phase_correlation"], sample_metrics["stedfm"]
            for key in sample_metrics.keys():
                results[key].append(sample_metrics[key]) 

            savename = os.path.basename(metadata["image_path"][0].replace(".tif", ""))
            if args.seed in [42, 97]:
                fig, axs = plt.subplots(1, 4, figsize=(20, 5))
                axs[0].imshow(confocal_np, cmap="hot", vmin=0, vmax=1)
                axs[1].imshow(sted_np, cmap="hot", vmin=0, vmax=1)
                axs[2].imshow(sample_np, cmap="hot", vmin=0, vmax=1)
                vmax_abs = float(np.max(np.abs(residual_image)))
                if vmax_abs <= 0:
                    vmax_abs = 1e-12
                res_norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)
                # bwr: blue (negative) -> white (0) -> red (positive)
                axs[3].imshow(residual_image, cmap="bwr", norm=res_norm)
                # axs[4].imshow(segmentation_rgb)
                axs[0].set_title("Confocal")
                axs[1].set_title("STED")
                axs[1].set_title(f"{sted_np.min():.4f} - {sted_np.max():.4f}")
                axs[2].set_title(f"{sample_np.min():.4f} - {sample_np.max():.4f}\n{residual_image.min():.4f} - {residual_image.max():.4f}")
                axs[3].set_title(f"MSE: {mse:.4f}\tPSNR: {psnr:.4f}\nSSIM: {ssim:.4f}\tMS-SSIM: {ms_ssim:.4f}\nFNCC: {fourier_ncc:.4f}\tW NCC: {wavelet_ncc:.4f}\nPCorr: {phase_correlation:.4f}\tSTEDFM: {stedfm_sim:.4f}")     
                # axs[4].set_title(f"Rings: {rings_dice:.4f}\nFibers: {fibers_dice:.4f}")    
                for ax in axs:
                    ax.axis("off")
                plt.tight_layout()
                fig.savefig(os.path.join(image_outdir, f"{savename}.png"), dpi=900, bbox_inches="tight")
                plt.close(fig)
            # tiff_data = np.stack(tiff_data, axis=0)
            
            # tifffile.imwrite(os.path.join(image_dir, f"{savename}_{args.model}_{subsample}-sample-{args.seed}.tif"), data=tiff_data.astype(np.float32))
    return results 



def bootstrap(data: np.ndarray, n_bootstraps: int = 100):
    num_samples = len(data)
    bootstrap_means = [] 
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=num_samples, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    return np.mean(bootstrap_means), np.std(bootstrap_means)

def analyze_results(pix2pix_results: dict, ddpm_results: dict, draft_results: dict, save_dir: str):
    real_resolution = np.load(os.path.join(save_dir, f"real_data_metrics.npz"))["resolution"]
    real_quality = np.load(os.path.join(save_dir, f"real_data_metrics.npz"))["quality"]
    for metric_key in METRIC_NAMES:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        ddpm_metric_avgs = [] 
        ddpm_metric_stds = [] 
        draft_metric_avgs = [] 
        draft_metric_stds = []
        pix2pix_metric_avgs = [] 
        pix2pix_metric_stds = [] 
        for subsample in tqdm(ddpm_results.keys(), desc="... Aggregating results ..."):
            # if subsample == None:
            #     subsample = "full"
            ddpm_data = ddpm_results[subsample][metric_key] 
            pix2pix_data = pix2pix_results[subsample][metric_key] 
            print(f"\n[---] Number of DDPM seeds for {subsample} subsample: {len(ddpm_data)} [---]")
            print(f"[---] Number of Pix2Pix seeds for {subsample} subsample: {len(pix2pix_data)} [---]")
            # ddpm_bootstrap_mean, ddpm_bootstrap_std = bootstrap(ddpm_data)
           
            ddpm_metric_avgs.append(np.mean(ddpm_data))
            ddpm_metric_stds.append(np.std(ddpm_data))
            pix2pix_metric_avgs.append(np.mean(pix2pix_data))
            pix2pix_metric_stds.append(np.std(pix2pix_data))
            try:
                draft_data = draft_results[subsample][metric_key] 
                print(f"[---] Number of DRaFT seeds for {subsample} subsample: {len(draft_data)} [---]")
                # draft_bootstrap_mean, draft_bootstrap_std = bootstrap(draft_data)
                draft_metric_avgs.append(np.mean(draft_data))
                draft_metric_stds.append(np.std(draft_data))
                # _, pvalue = mannwhitneyu(ddpm_data, draft_data) 
                # with open(os.path.join(save_dir, f"{metric_key}.txt"), "a") as f:
                #     f.write(f"{subsample}-{metric_key}: {pvalue:.5f}\n")
                # # print(f"\t{metric_key} p-value: {pvalue:.6f}")
            except:
                continue

        ddpm_metric_avgs = np.array(ddpm_metric_avgs)
        ddpm_metric_stds = np.array(ddpm_metric_stds)
        draft_metric_avgs = np.array(draft_metric_avgs)
        draft_metric_stds = np.array(draft_metric_stds)
        pix2pix_metric_avgs = np.array(pix2pix_metric_avgs)
        pix2pix_metric_stds = np.array(pix2pix_metric_stds)
        x = np.arange(len(ddpm_results.keys())) 
        x_draft = np.arange(len(draft_results.keys()))
        x_pix2pix = np.arange(len(pix2pix_results.keys()))
        ax.plot(x, ddpm_metric_avgs, color="tab:blue", marker='o', label="DDPM")
        ax.fill_between(x, ddpm_metric_avgs - ddpm_metric_stds, ddpm_metric_avgs + ddpm_metric_stds, color="tab:blue", alpha=0.2)
        ax.plot(x_pix2pix, pix2pix_metric_avgs, color="tab:green", marker='o', label="Pix2Pix")
        ax.fill_between(x_pix2pix, pix2pix_metric_avgs - pix2pix_metric_stds, pix2pix_metric_avgs + pix2pix_metric_stds, color="tab:green", alpha=0.2)
        ax.plot(x_draft, draft_metric_avgs, color="#CC503E", marker='o', label="DRaFT")
        ax.fill_between(x_draft, draft_metric_avgs - draft_metric_stds, draft_metric_avgs + draft_metric_stds, color="#CC503E", alpha=0.2)

        if metric_key == "resolution":
            ax.axhline(real_resolution, color="grey", linestyle="--", label="Real data")
        if metric_key == "quality":
            ax.axhline(real_quality, color="grey", linestyle="--", label="Real data")
        ax.set_xlabel("Subsample size")
        ax.set_ylabel(metric_key)
        ax.set_xticks(x)
        ax.set_xticklabels(list(ddpm_results.keys()))
        ax.legend()
        fig.savefig(os.path.join(save_dir, f"{metric_key}.pdf"), dpi=900, transparent=True, bbox_inches="tight")
        plt.close(fig)

def load_unet(checkpoint_unet: str, device: torch.device):
    unet = SegmentationUNet(in_channels=1, out_channels=2)
    unet.load_state_dict(torch.load(checkpoint_unet, map_location=device))
    unet.to(device)
    unet.eval()
    return unet

def compute_real_data_metrics(dataset: DendriticFActinDataset, quality_net: nn.Module):
    resolutions = []
    qualities = []
    for _, sted, _, _ in dataset:
        sted = sted.squeeze().cpu().numpy()
        resolution = compute_resolution(img=sted)
        quality = infer_quality(img=sted, quality_net=quality_net)
        resolutions.append(resolution)
        qualities.append(quality)
    return np.mean(resolutions), np.mean(qualities)

def main():
    LOG_FOLDER = os.path.join(f"./{args.dataset}-experiment/results/{args.model}") 
    
    
    os.makedirs(LOG_FOLDER, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        SEEDS = [9, 42, 87, 97, 99]
        # RESULTS_FOLDER = os.path.join(BASE_PATH, "baselines", "DRAFT", args.dataset, "results")
        RESULTS_FOLDER = f"./DendriticFActin-experiment/results"
        ddpm_results = {}
        draft_results = {}
        pix2pix_results = {} 
        for subsample in args.subsamples:
            if subsample == None:
                subsample = "full"
            seed_ddpm_results = defaultdict(list)
            seed_draft_results = defaultdict(list) 
            seed_pix2pix_results = defaultdict(list)
            for seed in SEEDS:
                ddpm_path = os.path.join(RESULTS_FOLDER, "DDPM", f"DDPM-{subsample}-sample-{seed}.npz") 
                draft_path = os.path.join(RESULTS_FOLDER, "DRAFT", f"DRAFT-{subsample}-sample-{seed}.npz")
                pix2pix_path = os.path.join(RESULTS_FOLDER, "Pix2Pix", f"Pix2Pix-{subsample}-sample-{seed}.npz")
                if os.path.exists(ddpm_path):
                    temp_ddim_results = np.load(ddpm_path)
                    temp_ddim_results = {key: np.array(temp_ddim_results[key]) for key in temp_ddim_results.keys()}
                    for key in temp_ddim_results.keys():
                        seed_ddpm_results[key].append(np.mean(temp_ddim_results[key]))
                else:
                    print(f"[---] No DDPM results found for {subsample}-sample-{seed} [---]")
                    
                if os.path.exists(draft_path):
                    temp_draft_results = np.load(draft_path)
                    temp_draft_results = {key: np.array(temp_draft_results[key]) for key in temp_draft_results.keys()}
                    for key in temp_draft_results.keys():
                        seed_draft_results[key].append(np.mean(temp_draft_results[key]))
                else:
                    print(f"[---] No DRAFT results found for {subsample}-sample-{seed} [---]")

                if os.path.exists(pix2pix_path):
                    temp_pix2pix_results = np.load(pix2pix_path)
                    temp_pix2pix_results = {key: np.array(temp_pix2pix_results[key]) for key in temp_pix2pix_results.keys()}
                    for key in temp_pix2pix_results.keys():
                        seed_pix2pix_results[key].append(np.mean(temp_pix2pix_results[key]))
                else:
                    print(f"[---] No Pix2Pix results found for {subsample}-sample-{seed} [---]")

            ddpm_results[subsample] = seed_ddpm_results
            draft_results[subsample] = seed_draft_results
            pix2pix_results[subsample] = seed_pix2pix_results

        analyze_results(pix2pix_results=pix2pix_results, ddpm_results=ddpm_results, draft_results=draft_results, save_dir=os.path.dirname(LOG_FOLDER))

    else:
        unet = load_unet(checkpoint_unet=args.unet_checkpoint, device=DEVICE)
        quality_net = load_quality_net()
        files = glob.glob(os.path.join(args.dataset_path, f"{args.dataset}Dataset", "test", "*.tif"))
        
        test_dataset = DendriticFActinDataset(files=files, split="test", coordinates_path=os.path.join(BASE_PATH, "Datasets", f"{args.dataset}Dataset-exported"))
        real_resolution, real_quality = compute_real_data_metrics(dataset=test_dataset, quality_net=quality_net) 
        np.savez(os.path.join(os.path.dirname(LOG_FOLDER), f"real_data_metrics.npz"), resolution=real_resolution, quality=real_quality)
        print(f"[---] Test dataset size: {len(test_dataset)} [---]")
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        missing_models = []
        for subsample in args.subsamples:
            if subsample == None:
                subsample = "full"
            RESULTS_FOLDER = os.path.join(BASE_PATH, "baselines", args.dataset, "results", f"{subsample}-sample")
            TIF_FOLDER = os.path.join(BASE_PATH, "baselines", args.dataset, "results", f"{args.model}-{subsample}-sample", "images")
            if os.path.exists(os.path.join(LOG_FOLDER, f"{args.model}-{subsample}-sample-{args.seed}.npz")):
                print(f"[---] Results already exist for {args.model}-{subsample}-sample-{args.seed}, skipping... [---]")
                continue
            os.makedirs(RESULTS_FOLDER, exist_ok=True)
            os.makedirs(TIF_FOLDER, exist_ok=True)
            try:
                model = load_models(model_type=args.model, subsample=subsample).to(DEVICE)
            except FileNotFoundError:
                missing_models.append(f"{args.model}-{subsample}-{args.seed}")
                continue

       
            results = inference(
                model=model, 
                dataloader=test_dataloader, 
                unet=unet,
                device=DEVICE, 
                save_dir=LOG_FOLDER, 
                image_dir=TIF_FOLDER,
                model_type=args.model, 
                subsample=subsample,
                quality_net=quality_net
            ) 
          
            np.savez(
                os.path.join(LOG_FOLDER, f"{args.model}-{subsample}-sample-{args.seed}.npz"),
                **results
            )

        if len(missing_models) > 0:
            for m in missing_models:
                print(f"\tMissing {m}")
            
                


if __name__ == "__main__":
    main()