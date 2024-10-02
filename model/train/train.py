import os
from time import time

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader

from model.train.dataset import XPSDataset
from model.model import XPSModel


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=0):     
        
        #flatten label and prediction tensors
        inputs = inputs
        targets = targets
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

ALPHA = 0.8
GAMMA = 2
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

ALPHA = 0.5
BETA = 0.5
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


def precision(inputs, targets):
    TP = (inputs * targets).sum()
    FP = (inputs * (1 - targets)).sum()
    return TP / (TP + FP)


def recall(inputs, targets):
    TP = (inputs * targets).sum()
    FN = (targets * (1-inputs)).sum()
    return TP / (TP + FN)


def accuracy(inputs, targets):
    TP = (inputs * targets).sum()
    TN = ((1 - inputs) * (1 - targets)).sum()
    FP = (inputs * (1 - targets)).sum()
    FN = (targets * (1-inputs)).sum()
    return (TP + TN) / (TP + TN + FP + FN)


def calc_metrics(inputs, targets, metrics={}):
    metrics['acc'].append(accuracy(inputs, targets))
    metrics['prec'].append(precision(inputs, targets))
    metrics['rec'].append(recall(inputs, targets))


def train_one_epoch(model, train_dataloader, test_dataloader, optimizer, criterion, metrics={}):
    mean_train_loss = 0
    mean_test_loss = 0

    model.train()
    for x, peak_mask, max_mask in train_dataloader:
        pred_peak_mask, pred_max_mask = model(x)
        
        gt = torch.cat((peak_mask, max_mask)).view(-1)
        pred = torch.cat((pred_peak_mask, pred_max_mask)).view(-1)
        
        train_loss = criterion(pred, gt)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        mean_train_loss += train_loss.detach().numpy()

    mean_train_loss /= len(train_dataloader)

    model.eval()
    with torch.no_grad():
        for x, peak_mask, max_mask in test_dataloader:
            pred_peak_mask, pred_max_mask = model(x)
            peak_targets = peak_mask.view(-1).detach().numpy()
            max_targets = max_mask.view(-1).detach().numpy()

            peak_inputs = pred_peak_mask.view(-1).detach().numpy()
            max_inputs = pred_max_mask.view(-1).detach().numpy()

            peak_loss = criterion(peak_inputs, peak_targets)
            max_loss = criterion(max_inputs, max_targets)

            mean_test_loss = peak_loss + max_loss
    
    mean_test_loss /= 2 * len(test_dataloader)
    calc_metrics(peak_inputs, peak_targets, metrics['peak'])
    calc_metrics(max_inputs, max_targets, metrics['max'])

    return mean_train_loss, mean_test_loss


#TODO: split loss count into peak and max
# def train(model, dataset_train, dataset_val, optimizer, metric=None, num_epochs=NUM_EPOCH):
#     iou_loss = IoULoss()
#     dice_loss = DiceLoss()
#     focal_loss = FocalLoss()
#     bce_loss = torch.nn.BCELoss()
#     tversky_loss = TverskyLoss()

#     dataloader_train = DataLoader(
#         dataset_train,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=os.cpu_count()
#     )
#     dataloader_val = DataLoader(dataset_val, shuffle=True)

#     metrics_dict = {
#         'train_loss': [],
#         'val_loss': [],
#         'dice_loss': [],
#         'accuracy': [],
#         'precision': [],
#         'recall': []
#     }

#     for epoch in range(num_epochs):
#         start_time = time()
#         model.train()

#         train_loss = 0
#         val_loss = 0

#         for x, peak_mask, max_mask in dataloader_train:
#             pred_peak_mask = model(x)
#             gt = max_mask.view(-1)
#             pred = pred_peak_mask.view(-1)
#             # gt = torch.cat((peak_mask)).view(-1)

#             # pred = torch.cat((pred_peak_mask)).view(-1)

#             loss = iou_loss(pred, gt)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             train_loss += metric(pred, gt).detach().numpy()

#         model.eval()
#         with torch.no_grad():
#             for x, peak_mask, max_mask in dataloader_val:
#                 pred_peak_mask = model(x)
#                 gt = max_mask.view(-1)
#                 pred = pred_peak_mask.view(-1)
#                 # gt = torch.cat((peak_mask)).view(-1)

#                 # pred = torch.cat((pred_peak_mask)).view(-1)

#                 val_loss += metric(pred, gt).detach().numpy()
        
#         end_time = time()

#         train_loss = train_loss / len(dataloader_train)
#         val_loss = val_loss / len(dataloader_val)
           

#         if best_loss > val_loss:
#             best_weights = model.state_dict()
#             best_loss = val_loss
        
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

#         if epoch % 10 == 0:
#             torch.save(best_weights, f'train_log/last_{best_loss:.3f}')
#             print(f'Epoch: {epoch} finished at {end_time - start_time}')
#             print(f'Train loss: {train_loss}; Validation loss {val_loss}\n')
#             print(f'') 

#     return best_weights, train_losses, val_losses


if __name__ == '__main__':
    NUM_EPOCH = 50
    BATCH_SIZE = 500

    model = XPSModel()
    # model.load_state_dict(torch.load('weights/v0.5_best'))

    train_dataloader = DataLoader(
        XPSDataset('data/data_to_train'),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count()
    )

    test_dataloader = DataLoader(
        XPSDataset('data/data_to_test'),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count()
    )

    # 5e-4 fast converge for iou and tversky
    optimizer = Adam(model.parameters(), lr=5e-4)

    # optimizer = RMSprop(model.parameters(), lr=1e-4)
    criterion = IoULoss()
    
    metrics = {
        'peak': {'acc': [],
                 'prec': [],
                 'rec': []},
        'max': {'acc': [],
                 'prec': [],
                 'rec': []}
    }
    train_losses, test_losses = [], []

    start = time()
    for epoch in range(NUM_EPOCH):
        train_loss, test_loss = train_one_epoch(model, train_dataloader, test_dataloader, optimizer, criterion, metrics)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 5 == 0:
            print(f'Epoch: {epoch}/{NUM_EPOCH} end at {time() - start:.1f} s\n' + 
                  f'Train loss: {train_loss}, Test loss: {test_loss}')

    plt.plot(np.arange(NUM_EPOCH), train_losses, test_losses)
    plt.show()
    # plt.plot(
    #     np.arange(NUM_EPOCH),
    #     metrics['peak']['acc'],
    #     metrics['peak']['prec'],
    #     metrics['peak']['rec'],
    #     metrics['max']['acc'],
    #     metrics['max']['prec'],
    #     metrics['max']['rec'],
    # )
    plt.show()

    # plt.plot(np.arange(NUM_EPOCH), train_losses, val_losses)

    # torch.save(best_weights, 'weights/v0.9_max_mask_iou')
    # train_one_epoch(model, dataloader_train, optimizer, iou_loss)
