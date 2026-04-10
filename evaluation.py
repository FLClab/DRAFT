from re import L
import numpy as np 
import torch 
import argparse

from metrics import compute_metrics 
import tifffile 
from torch import nn
from banditopt.objectives import Resolution 
from tqdm import tqdm
import glob
from typing import List, Dict
from QualityNet.networks import NetTrueFCN 
import os
import matplotlib.pyplot as plt
from stedfm.DEFAULTS import BASE_PATH
import pickle

SUBSAMPLES = [str(item) for item in [50, 100, 250, 300, 500, 1000, 2000, 3000]] + ["full"]
MODELS = ["Pix2Pix", "DDPM", "DRAFT"]
BASE_PATH = "/home-local/Frederic/baselines/DRAFT/DendriticFActin/samples"
SEEDS = [9, 42, 87, 97, 99]
METRICS = ["mae", "mse", "psnr", "ssim", "ms_ssim", "resolution", "quality"]
UNET_CHECKPOINT = os.path.join(BASE_PATH, "baselines", "pretrained-unet-01", "params.net")

def load_quality_net() -> nn.Module:
    quality_net = NetTrueFCN()
    quality_checkpoint = torch.load(f"./QualityNet/trained_models/actin/qualitynet.pth", weights_only=False)
    quality_net.load_state_dict(quality_checkpoint["model_state_dict"])
    return quality_net

def compute_microscopy_metrics(image: np.ndarray) -> dict:
    pass

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

def process_subsample(model: str, subsample: str, seed: int):
    files = glob.glob(os.path.join(BASE_PATH, f"{model}-{subsample}-sample", "images", f"*-{str(seed)}.tif"))
    files = list(set(files))
    return files

def process_files(files: List, quality_net: nn.Module, device: torch.device): 
    scores = {metric_key: [] for metric_key in METRICS}
    for file in files:
        img = tifffile.imread(file)
        confocal, sted, gen = img[0], img[1], img[2]
        sample_metrics = compute_metrics(truth_image=sted, prediction_image=gen)
        quality = infer_quality(img=gen, quality_net=quality_net)
        resolution = compute_resolution(img=gen)
        for metric in sample_metrics.keys():
            scores[metric].append(sample_metrics[metric])

        scores["quality"].append(quality)
        scores["resolution"].append(resolution) 
    return scores

def plot_results(model_avgs: Dict, model_errs : Dict, real_resolution: float, real_quality: float):
    outpath = "/home/frederic/DRAFT/DendriticFActin-experiment"
    for metric in METRICS:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111) 
        pix2pix_avg = np.array(model_avgs["Pix2Pix"][metric])
        pix2pix_err = np.array(model_errs["Pix2Pix"][metric])
        ddpm_avg = np.array(model_avgs["DDPM"][metric])
        ddpm_err = np.array(model_errs["DDPM"][metric])
        draft_avg = np.array(model_avgs["DRAFT"][metric])
        draft_err = np.array(model_errs["DRAFT"][metric])
        print(pix2pix_avg.shape, ddpm_avg.shape, draft_avg.shape)
        ax.plot(np.arange(len(pix2pix_avg)), pix2pix_avg, color="tab:green", marker='o', label="Pix2Pix")
        ax.fill_between(pix2pix_avg, pix2pix_avg - pix2pix_err, pix2pix_avg + pix2pix_err, color="tab:green", alpha=0.2)
        ax.plot(np.arange(len(ddpm_avg)), ddpm_avg, color="tab:blue", marker='o', label="DDPM")
        ax.fill_between(ddpm_avg, ddpm_avg - ddpm_err, ddpm_avg + ddpm_err, color="tab:blue", alpha=0.2)
        ax.plot(np.arange(len(draft_avg)), draft_avg, color="#CC503E", marker='o', label="DRAFT")
        ax.fill_between(draft_avg, draft_avg - draft_err, draft_avg + draft_err, color="#CC503E", alpha=0.2)
        if metric == "resolution":
            ax.axhline(real_resolution, color="grey", linestyle="--", label="Real data")
        if metric == "quality":
            ax.axhline(real_quality, color="grey", linestyle="--", label="Real data")
        ax.set_xlabel("Subsample size")
        ax.set_ylabel(metric)
        ax.set_xticks(np.arange(len(SUBSAMPLES)))
        ax.set_xticklabels(SUBSAMPLES)
        ax.legend()
        fig.savefig(f"{outpath}/{metric}.pdf", dpi=900, transparent=True, bbox_inches="tight")
        plt.close(fig)

def compute_real_data_metrics(files: List, quality_net: nn.Module):
    real_resolution = [] 
    real_quality = []
    for f in files:
        real_img = tifffile.imread(f)[1]
        real_resolution.append(compute_resolution(img=real_img))
        real_quality.append(infer_quality(img=real_img, quality_net=quality_net))
    return real_resolution, real_quality

def main():
    DEVICE = torch.device("cpu")
    outpath = "/home/frederic/DRAFT/DendriticFActin-experiment"
    if not os.path.exists(os.path.join(outpath, "model_avgs.pkl")):
        os.makedirs(outpath, exist_ok=True)
        quality_net = load_quality_net()
        model_avgs = {
            model_key: {
                metric_key: [] for metric_key in METRICS
            } for model_key in MODELS
        }
        model_errs = {
            model_key: {
                metric_key: [] for metric_key in METRICS
            } for model_key in MODELS
        }

        
        for model in tqdm(MODELS, desc="... Processing models ..."):
            seed_scores = {
                metric_key: np.zeros((len(SEEDS), len(SUBSAMPLES))) for metric_key in METRICS
            }
            for seed_idx, seed in enumerate(tqdm(SEEDS, desc="... Processing seeds ...")):
                for subsample_idx, subsample in enumerate(tqdm(SUBSAMPLES, desc="... Processing subsamples ...")):
                    files = process_subsample(model, subsample, seed)
                    scores = process_files(files=files, quality_net=quality_net, device=DEVICE)
                    for metric in scores.keys():
                        seed_scores[metric][seed_idx, subsample_idx] = np.mean(scores[metric])

            for metric in model_avgs[model].keys():
                seed_avg = np.mean(seed_scores[metric], axis=0)
                seed_std = np.std(seed_scores[metric], axis=0)
                print(seed_avg.shape, seed_std.shape)
                model_avgs[model][metric] = (seed_avg)
                model_errs[model][metric] = (seed_std)
                       
        
    

        real_resolution, real_quality = compute_real_data_metrics(files=files, quality_net=quality_net)
        real_resolution = np.mean(real_resolution)
        real_quality = np.mean(real_quality)

        np.savez(os.path.join(outpath, "real_data_metrics.npz"), real_resolution=real_resolution, real_quality=real_quality)

        with open(os.path.join(outpath, "model_avgs.pkl"), "wb") as f:
            pickle.dump(model_avgs, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(outpath, "model_errs.pkl"), "wb") as f:
            pickle.dump(model_errs, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    else:
        with open(os.path.join(outpath, "model_avgs.pkl"), "rb") as f:
            model_avgs = pickle.load(f)
        with open(os.path.join(outpath, "model_errs.pkl"), "rb") as f:
            model_errs = pickle.load(f)
        real_resolution = np.load(os.path.join(outpath, "real_data_metrics.npz"))["real_resolution"]
        real_quality = np.load(os.path.join(outpath, "real_data_metrics.npz"))["real_quality"]

    
    plot_results(
        model_avgs=model_avgs, 
        model_errs=model_errs, 
        real_resolution=real_resolution,
        real_quality=real_quality
    )


if __name__=="__main__":
    main()