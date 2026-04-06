import numpy as np 
import tifffile 
import torch 
from torch import nn
import matplotlib.pyplot as plt 
import argparse 
from tiffwrapper import make_composite 
from stedfm.DEFAULTS import BASE_PATH 
from stedfm import get_pretrained_model_v2 
from torch.utils.data import DataLoader 
import os 
import random
from typing import List
from tqdm import tqdm, trange 
import torchvision.transforms as T
import glob 
from denoising_unet import UNet
from diffusion_model import DDPM 
from datasets.dendrites_dataset import DendriticFActinDataset 
from datasets.axons_dataset import AxonalRingsDataset 
from utils import AverageMeter, SaveBestModel

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, default="DendriticFActinDataset")
parser.add_argument("--dataset-path", type=str, default=os.path.join(BASE_PATH, "Datasets"))
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--save-folder", type=str, default=os.path.join(BASE_PATH, "baselines", "DRAFT", "DendriticFActin"))
parser.add_argument("--subsample", type=int, default=None)
args = parser.parse_args()


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    set_seeds(args.seed)
    

if __name__=="__main__":
    main()
