import numpy as np 
import matplotlib.pyplot as plt
from stedfm.DEFAULTS import BASE_PATH 
import os 
import argparse 
from tqdm import tqdm 
from metrics import compute_fourier_ncc_bands
from itertools import combinations
from scipy.stats import mannwhitneyu

SUBSAMPLES = [50, 100, 300, 500, 1000, 2000, 3000, "full"]
MODELS = ["Pix2Pix", "DDPM", "DRAFT"]
SEEDS = [9, 42, 87, 97, 99]

def compute_stats(data: dict, save_dir: str):
    os.makedirs(os.path.join(save_dir, "stats"), exist_ok=True)
    for i in range(len(SUBSAMPLES)):
        fourier_data = [data[i] for data in data.values()]
        fourier_data = {"Pix2Pix": fourier_data[0], "DDPM": fourier_data[1], "DRAFT": fourier_data[2]}
        pairs = combinations(fourier_data.keys(), 2)
        out_path = os.path.join(save_dir, "stats", f"fourier_stats_{SUBSAMPLES[i]}_sample.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for (p1, p2) in pairs:
                

                data_p1 = [float(item) for item in fourier_data[p1]]
                data_p2 = [float(item) for item in fourier_data[p2]]
                w, p = mannwhitneyu(data_p1, data_p2)
                f.write(f"{p1} vs {p2}: w={w:.4f}, p={p:.6f}\n")


def main():
    OUTPUT_DIR = f"./DendriticFActin-experiment/Fourier-Space"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    low_freq_ncc = {model: [] for model in MODELS}
    high_freq_ncc = {model: [] for model in MODELS}
    low_freq_err = {model: [] for model in MODELS}
    high_freq_err = {model: [] for model in MODELS}
    stats = {model: [] for model in MODELS}
    y = {model: [] for model in MODELS}
    for subsample in tqdm(SUBSAMPLES, desc=" ... Processing subsamples ..."):
        for model in MODELS:
            seed_low_ncc = []
            seed_high_ncc = []
            for seed in SEEDS:
                
                path = os.path.join(BASE_PATH, "baselines", "DRAFT", "DendriticFActin", "results", model, f"{model}-{subsample}-sample-{seed}.npz")
                if not os.path.exists(path):
                    continue 
               
                data = np.load(path)
                data = {key: np.array(data[key]) for key in data.keys()} 
                sted_images = data["steds"]
                sample_images = data["samples"]
                images_low = [] 
                images_high = []
                for pred, truth in zip(sample_images, sted_images):
                    low_ncc, high_ncc = compute_fourier_ncc_bands(pred, truth, low_freq_ratio=0.142)
                    images_low.append(low_ncc)
                    images_high.append(high_ncc)

                seed_low_ncc.append(np.mean(images_low))
                seed_high_ncc.append(np.mean(images_high))
            
            low_arr = np.array(seed_low_ncc)
            high_arr = np.array(seed_high_ncc)
            stats[model].append(seed_high_ncc)
            low_freq_ncc[model].append(np.mean(low_arr))
            high_freq_ncc[model].append(np.mean(high_arr))
            low_n = low_arr.size 
            high_n = high_arr.size 
            low_sem = np.std(low_arr, ddof=1) / np.sqrt(low_n)
            high_sem = np.std(high_arr, ddof=1) / np.sqrt(high_n)
            
            low_freq_err[model].append(low_sem)
            high_freq_err[model].append(high_sem)


    compute_stats(data=stats, save_dir=OUTPUT_DIR)
    
    fig1 = plt.figure(figsize=(3,3))
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(3,3))
    ax2 = fig2.add_subplot(111)
    x = np.arange(len(SUBSAMPLES))
    for model, color in zip(MODELS, ["tab:green", "tab:blue", "#CC503E"]): 
        y_low = np.array(low_freq_ncc[model])
        y_high = np.array(high_freq_ncc[model])
        y_low_err = np.array(low_freq_err[model])
        y_high_err = np.array(high_freq_err[model])
        ax1.plot(x, y_low, color=color, label=f"{model}", marker='o')
        ax1.fill_between(x, y_low - y_low_err, y_low + y_low_err, color=color, alpha=0.2)
        ax2.plot(x, y_high, color=color, label=f"{model}", marker='o')
        ax2.fill_between(x, y_high - y_high_err, y_high + y_high_err, color=color, alpha=0.2)
    for ax in [ax1, ax2]:
        ax.set_xlabel("Subsample size")
        ax.set_ylabel("Fourier NCC")
        ax.set_xticks(x)
        ax.set_xticklabels(SUBSAMPLES)
    ax1.legend()
    ax1.set_title("Low-Frequency NCC")
    ax2.set_title("High-Frequency NCC")

    fig1.set_size_inches(3,3)
    fig2.set_size_inches(3,3)
    fig1.savefig(os.path.join(OUTPUT_DIR, "ncc_low_bands.pdf"), transparent=True, dpi=900)
    fig2.savefig(os.path.join(OUTPUT_DIR, "ncc_high_bands.pdf"), transparent=True, dpi=900)
    plt.close(fig1)
    plt.close(fig2)

                
                   
    

if __name__=="__main__":
    main()