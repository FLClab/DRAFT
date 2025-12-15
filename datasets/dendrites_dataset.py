from torch.utils.data import Dataset 
from typing import List, Tuple, Optional, Callable
import tifffile 
import torch 
import numpy as np
import glob
import os

class DendriticFActinDataset(Dataset):
    def __init__(
            self,
            files: List[str],
            transform: Optional[Callable] = None,
            split: str = "train",
            coordinates_path: Optional[str] = None,
    ) -> None:
        self.files = files 
        self.transform = transform
        self.split = split
        if self.split == "test":
            self.coordinates = glob.glob(os.path.join(coordinates_path, "test", "high_resolution", "*.tif"))
            print(f"Found {len(self.coordinates)} coordinates")
        else:
            self.coordinates = None

    def __len__(self) -> int:
        return len(self.files)

    def get_coordinates(self, fname: str) -> Tuple[np.ndarray, np.ndarray]:
        found = [fname == os.path.basename(f).split(".")[0] for f in self.coordinates]
        if any(found):
            file = self.coordinates[found.index(True)] 
            coordinates = os.path.basename(file).split(".")[-2].split("_")
            y, x = int(coordinates[-2]), int(coordinates[-1])
            return y, x 
        else:
            raise ValueError(f"Coordinates not found for {fname}")
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        file = self.files[idx]
        fname = os.path.basename(file).split(".")[0]
        data = tifffile.imread(file)


        if self.split == "test":
            y, x = self.get_coordinates(fname)
            confocal, sted = data[0, y:y+224, x:x+224], data[1, y:y+224, x:x+224]
            rings, fibers = data[2, y:y+224, x:x+224], data[3, y:y+224, x:x+224]
            assert confocal.shape == (224, 224)
            assert sted.shape == (224, 224)
            assert rings.shape == (224, 224)
            assert fibers.shape == (224, 224)
        else:    
            confocal, sted = data[0, :, :], data[1, :, :]
            rings, fibers = data[2, :, :], data[3, :, :]
        
        if rings.max() > 1:
            rings = (rings / rings.max())
        if fibers.max() > 1:
            fibers = (fibers / fibers.max())

        rings = rings.astype(np.uint8)
        fibers = fibers.astype(np.uint8)
        m_conf, M_conf = np.min(confocal), np.max(confocal)
        m_sted, M_sted = np.min(sted), np.max(sted)
        confocal = (confocal - m_conf) / (M_conf - m_conf)
        sted = (sted - m_sted) / (M_sted - m_sted)
        # confocal = np.clip(confocal / 255.0, 0, 1)
        # sted = np.clip(sted / 255.0, 0, 1)
        
        confocal = torch.tensor(confocal[np.newaxis, ...], dtype=torch.float32)
        sted = torch.tensor(sted[np.newaxis, ...], dtype=torch.float32)
        rings = torch.tensor(rings[np.newaxis, ...], dtype=torch.float32)
        fibers = torch.tensor(fibers[np.newaxis, ...], dtype=torch.float32)
        ground_truth = torch.cat([rings, fibers], dim=0)
        metadata = {"image_path": file, "y": y, "x": x}
        return confocal, sted, ground_truth, metadata