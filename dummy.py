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
parser.add_argument("--dataset", type=str, default="DendriticFActinDataset")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets"))
args = parser.parse_args()

if __name__=="__main__":
    os.makedirs("./dendrites-examples", exist_ok=True)
    files = glob.glob(os.path.join(args.dataset_path, args.dataset, "train", "*.tif")) 
    np.random.shuffle(files)
    dataset = DendriticFActinDataset(files=files, transform=None) 
    
    for i, (confocal, sted, ground_truth, _) in tqdm(enumerate(dataset), desc=" ... Processing dataset ...", total=50):
        confocal = confocal.squeeze().numpy()
        sted = sted.squeeze().numpy()
        ground_truth = ground_truth.squeeze().numpy() 
        segmentation = make_composite(ground_truth, luts=['green', "magenta"], ranges=[(0, 1), (0, 1)])
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(confocal, cmap="hot", vmin=0, vmax=1)
        axs[1].imshow(sted, cmap="hot", vmin=0, vmax=1)
        axs[2].imshow(segmentation)
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(f"./dendrites-examples/sample_{i}.pdf", transparent=True, dpi=900, bbox_inches="tight")
        plt.close(fig)
        if i >= 50:
            break

