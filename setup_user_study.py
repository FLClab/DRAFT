import numpy as np 
import matplotlib.pyplot as plt 
import argparse 
from collections import defaultdict 
import os
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument("--subsample", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

def main():
    MODELS = ["Pix2Pix"]#, "DDPM", "DRAFT"]
    RESULTS_FOLDER = f"/home-local/Frederic/baselines/DRAFT/DendriticFActin/results"
    TEMPLATES_FOLDER = f"./static/test/templates/{args.subsample if args.subsample is not None else 'full'}-sample"
    CANDIDATES_FOLDER = f"./static/test/candidates/{args.subsample if args.subsample is not None else 'full'}-sample"

    os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
    os.makedirs(CANDIDATES_FOLDER, exist_ok=True)

    for i, model in enumerate(MODELS):
        data = np.load(os.path.join(RESULTS_FOLDER, model, f"{model}-{args.subsample if args.subsample is not None else 'full'}-sample-{args.seed}.npz"))
       
        confocals = data["confoncals"]
        steds = data["steds"] 
        samples = data["samples"]
        assert confocals.shape[0] == steds.shape[0] == samples.shape[0] == 26 
        for img_idx in trange(confocals.shape[0], desc=f"... Processing {model} ..."):
            if i == 0:
                sted = steds[img_idx] 
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(sted, vmin=0, vmax=1, cmap="hot")
                ax.axis("off")
                fig.savefig(os.path.join(TEMPLATES_FOLDER, f"sted_{img_idx}.png"), dpi=600, bbox_inches="tight")
                plt.close(fig)
            
            sample = samples[img_idx]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(sample, vmin=0, vmax=1, cmap="hot")
            ax.axis("off")
            fig.savefig(os.path.join(CANDIDATES_FOLDER, f"{model.lower()}_{img_idx}.png"), dpi=600, bbox_inches="tight")
            plt.close(fig)



if __name__=="__main__":
    main()