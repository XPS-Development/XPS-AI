import numpy as np
import torch
from torch.utils.data import DataLoader

from model import XPSModel
from dataset import XPSDataset
from utils import view_labeled_data


PATH_TO_VAL = 'data/data_to_val'
TRESHOLD = 0.5

model = XPSModel()
model.load_state_dict(torch.load('WEIGHTS'))
model.eval()

dataset_val = XPSDataset(PATH_TO_VAL)
dataloader_val = DataLoader(dataset_val, shuffle=False)


for x, peak_mask, max_mask in dataloader_val:
    pred_peak_mask, pred_max_mask = model(x)
    y = x.view(-1).detach().cpu().numpy()
    x = np.arange(1, 257, step=1)

    pred_peak_mask = pred_peak_mask.view(-1).detach().cpu().numpy()
    pred_peak_mask = (pred_peak_mask > TRESHOLD)

    pred_max_mask = pred_max_mask.view(-1).detach().cpu().numpy()
    pred_max_mask = (pred_max_mask > TRESHOLD)

    view_labeled_data(x, y, (pred_peak_mask, pred_max_mask))
    