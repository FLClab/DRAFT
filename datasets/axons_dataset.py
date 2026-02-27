import numpy as np
import tifffile 
import torch
import os 
import glob 
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable

class AxonalRingsDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file = self.files[idx]
        data = tifffile.imread(file)
        # confocal, sted, rings = data[0] / 255.0, data[1] / 255.0, data[2] / 255.0
        confocal, sted, rings = data[0], data[1], data[2] 
        m_conf, M_conf = np.min(confocal), np.max(confocal)
        m_sted, M_sted = np.min(sted), np.max(sted)
        confocal = (confocal - m_conf) / (M_conf - m_conf)
        sted = (sted - m_sted) / (M_sted - m_sted) 
        if rings.max() != 0:
            rings = (rings / rings.max()).astype(np.uint8)
        confocal = torch.tensor(confocal[np.newaxis, ...], dtype=torch.float32)
        sted = torch.tensor(sted[np.newaxis, ...], dtype=torch.float32)
        rings = torch.tensor(rings[np.newaxis, ...], dtype=torch.float32)
        metadata = {"image_path": file}
        return confocal, sted, rings, metadata
        

# if __name__=="__main__":
#     files = glob.glob("/home-local/Frederic/Datasets/AxonalRingsDataset/train/*.tif")
#     dataset = AxonalRingsDataset(files=files)
#     print(len(dataset))
#     for i in range(len(dataset)):
#         confocal, sted, rings, metadata = dataset[i]
#         print(confocal.shape, sted.shape, rings.shape)
#         print(sted.max(), sted.min())
#         print("\n")