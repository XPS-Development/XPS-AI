from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class XPSDataset(Dataset):
    def __init__(self, path, device='cpu'):
        super().__init__()

        self.device = torch.device(device)
        self.data = []

        for f in Path(path).iterdir():
            array = np.loadtxt(f, delimiter=',')
            self.data.append(array)
        
    def __getitem__(self, index):
        array = self.data[index]

        x = array[:, 0]
        x = torch.tensor(x ,dtype=torch.float32, device=self.device).view(1, -1)
        x = (x - x.min()) / (x.max() - x.min())

        peak_mask = array[:, 1]
        peak_mask = torch.tensor(peak_mask, dtype=torch.float32, device=self.device)

        max_mask = array[:, 2]
        max_mask = torch.tensor(max_mask, dtype=torch.float32, device=self.device)
        
        return x, peak_mask, max_mask
    
    def __len__(self):
        return len(self.data)        
