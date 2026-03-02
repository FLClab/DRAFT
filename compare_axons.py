import numpy as np
import torch 
import argparse 
from stedfm.DEFAULTS import BASE_PATH 
import os 
from torch import nn 
from stedfm import get_pretrained_model_v2 
from denoising_unet import UNet 
from diffusion_model_draft import DRaFT_DDPM, RewardEncoder 
from datasets.axons_dataset import AxonalRingsDataset 
from torch.utils.data import DataLoader 
from typing import List 
from tqdm import tqdm, trange 
import glob 
import matplotlib.pyplot as plt 
from metrics import compute_metrics