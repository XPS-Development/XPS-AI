import os
from time import time

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn

from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import XPSDataset
from model import XPSModel


PATH_TO_TRAIN = 'data/data_to_train'
PATH_TO_VAL = 'data/data_to_val'
# PATH_TO_VAL = PATH_TO_TRAIN
BATCH_SIZE = 5
NUM_EPOCH = 100

class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=0):     
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


def train(model, dataloader_train, dataloader_val, optimizer, criterion, metric=None, num_epochs=NUM_EPOCH):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        start_time = time()
        model.train()

        train_loss = 0
        val_loss = 0

        for x, peak_mask, max_mask in dataloader_train:
            pred_peak_mask, pred_max_mask = model(x)
            gt = torch.cat((peak_mask, max_mask)).view(-1)

            pred = torch.cat((pred_peak_mask, pred_max_mask)).view(-1)

            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += metric(pred, gt)
        
        model.eval()
        for x, peak_mask, max_mask in dataloader_val:
            pred_peak_mask, pred_max_mask = model(x)
            gt = torch.cat((peak_mask, max_mask)).view(-1)

            pred = torch.cat((pred_peak_mask, pred_max_mask)).view(-1)

            val_loss += metric(pred, gt)
        
        end_time = time()
        
        print(f'Epoch: {epoch} finished at {end_time - start_time}')
        print(f'Train loss: {train_loss / len(dataset_train)}; Validation loss {val_loss / len(dataset_val)}\n')
        train_losses.append(train_loss.detach().numpy() / len(dataset_train))
        val_losses.append(val_loss.detach().numpy() / len(dataset_val))

    plt.plot(np.arange(num_epochs), train_losses, val_losses)
    plt.show()

    return model


if __name__ == '__main__':

    model = XPSModel()

    iou_loss = IoULoss()

    dataset_train = XPSDataset(PATH_TO_TRAIN)
    dataset_val = XPSDataset(PATH_TO_VAL)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count()
    )
    dataloader_val = DataLoader(dataset_val, shuffle=True)

    optimizer = Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.BCELoss(size_average=True)   

    model = train(model, dataloader_train, dataloader_val, optimizer, criterion, metric=iou_loss)
    torch.save(model.state_dict(), 'WEIGHTS')
