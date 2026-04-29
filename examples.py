import numpy as np 
import matplotlib.pyplot as plt 
from datasets.dendrites_dataset import DendriticFActinDataset 
import argparse 
import os 
from stedfm.DEFAULTS import BASE_PATH
import glob 
from tqdm import tqdm 
from tiffwrapper import make_composite

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="/home-local/Frederic/baselines/DRAFT/DendriticFActin/results")
parser.add_argument("--model", type=str, default="DRAFT")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if __name__=="__main__":
    SUBSAMPLES = [2000] #[50, 100, 300, 500, 1000, 2000, 3000, "full"]
    OUTDIR = f"./metric-examples/{args.model}"
    os.makedirs(OUTDIR, exist_ok=True)

    for subsample in tqdm(SUBSAMPLES, desc=" ... Processing subsamples ..."):
        os.makedirs(os.path.join(OUTDIR, f"{subsample}"), exist_ok=True)
        data = np.load(os.path.join(args.root, args.model, f"{args.model}-{subsample}-sample-{args.seed}.npz"))
        confocals = data["confoncals"]
        steds = data["steds"]
        samples = data["samples"]
        mses = data["mse"]
        ssims = data["ssim"]
        assert confocals.shape[0] == steds.shape[0] == samples.shape[0] == 26  
        for img_idx in tqdm(range(confocals.shape[0]), desc=" ... Processing images ..."):
            conf, sted, sample = confocals[img_idx], steds[img_idx], samples[img_idx]
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(conf, cmap="hot", vmin=0, vmax=1)
            axs[1].imshow(sted, cmap="hot", vmin=0, vmax=1)
            axs[2].imshow(sample, cmap="hot", vmin=0, vmax=1)
            axs[2].set_title(f"MSE: {mses[img_idx]:.4f}, SSIM: {ssims[img_idx]:.4f}")
            for ax in axs:
                ax.axis("off")
            plt.tight_layout()
            fig.savefig(os.path.join(OUTDIR, f"{subsample}", f"sample_{img_idx}.pdf"), transparent=True, dpi=900, bbox_inches="tight")
            plt.close(fig)
    





