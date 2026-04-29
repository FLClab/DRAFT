from re import L
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
from torch import nn 
import tifffile  
import argparse 
from stedfm.DEFAULTS import BASE_PATH
import os 
from tqdm import tqdm
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--x-metric", type=str, default="mse")
parser.add_argument("--y-metric", type=str, default="stedfm")
args = parser.parse_args()


SUBSAMPLES = [50, 100, 300, 500, 1000, 2000, 3000, "full"]

def main():
    OUTPUT_DIR = f"./DendriticFActin-experiment/correlation"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    RESULTS_FOLDER = os.path.join(BASE_PATH, "baselines", "DRAFT", "DendriticFActin", "results")
    x = {model: [] for model in ["Pix2Pix", "DDPM", "DRAFT"]}
    y = {model: [] for model in ["Pix2Pix", "DDPM", "DRAFT"]}
    for subsample in tqdm(SUBSAMPLES, desc=" ... Processing subsamples ..."):
        for model in ["Pix2Pix", "DDPM", "DRAFT"]:
            for seed in [9, 42, 87, 97, 99]:
                path = os.path.join(RESULTS_FOLDER, model, f"{model}-{subsample}-sample-{seed}.npz")
                if os.path.exists(path):
                    data = np.load(path)
                    data = {key: np.array(data[key]) for key in data.keys()} 
                    x[model].extend(data[args.x_metric].tolist())
                    y[model].extend(data[args.y_metric].tolist())
                    # x[model].append(np.mean(data[args.x_metric]))
                    # y[model].append(np.mean(data[args.y_metric]))

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111) 
    for model, color in zip(["Pix2Pix", "DDPM", "DRAFT"], ["tab:green", "tab:blue", "#CC503E"]): 
        r, p = spearmanr(x[model], y[model])
        print(f"{model}: r={r:.4f}, p={p:.4f}")
        ax.scatter(x[model], y[model], color=color, label=f"{model} (r={r:.4f})", s=2, alpha=0.4)
    ax.set_xlabel(args.x_metric)
    ax.set_ylabel(args.y_metric)
    ax.legend()
    ax.set_xscale("log")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{args.x_metric}-{args.y_metric}.pdf"), transparent=True, dpi=900)
    plt.close(fig)

    



if __name__=="__main__":
    main()