import numpy as np 
import matplotlib.pyplot as plt 
import argparse 
from collections import defaultdict
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Pix2Pix")
parser.add_argument("--seeds", nargs="+", type=int, default=[9, 42, 87, 97, 99])
parser.add_argument("--subsamples", nargs="+", type=int, default=[50, 100, 250, 300, 500, 1000, 2000, 3000])
args = parser.parse_args()



def main():
    RESULTS_FOLDER = f"./DendriticFActin-experiment/results/{args.model}"
    os.makedirs(os.path.join(RESULTS_FOLDER, "seeds"), exist_ok=True)
    for metric in ["mse", "mae", "psnr", "ssim", "ms_ssim", "rings_dice", "fibers_dice", "fourier_ncc", "wavelet_ncc", "phase_correlation", "stedfm"]:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        for seed in args.seeds:
    
            results = []
            
            for subsample in args.subsamples:
                if subsample == None:
                    subsample = "full"
                data = np.load(os.path.join(RESULTS_FOLDER, f"{args.model}-{subsample}-sample-{seed}.npz"))
                data = np.array(data[metric]) 
                results.append(np.mean(data))

            ax.plot(np.arange(len(args.subsamples)), results, marker='.', label=f"Seed {seed}")
        ax.set_xticks(np.arange(len(args.subsamples)))
        ax.set_xticklabels(args.subsamples[:-1] + ["Full"], rotation=45)
        ax.set_xlabel("Subsample size")
        ax.set_ylabel(metric)
        ax.legend()
        fig.savefig(os.path.join(RESULTS_FOLDER, "seeds", f"{metric}.pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)

            

                
         

if __name__ == "__main__":
    main()