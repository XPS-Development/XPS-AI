import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class IoULoss(nn.Module):
    def __init__(self, smooth=0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):     

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + self.smooth)/(union + self.smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE

        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, inputs, targets):

        TP = (inputs * targets).sum()
        TN = ((1 - inputs) * (1 - targets)).sum()
        FP = (inputs * (1 - targets)).sum()
        FN = (targets * (1-inputs)).sum()

        return (TP + TN) / (TP + TN + FP + FN)


class Precision(nn.Module):
    def __init__(self):
        super(Precision, self).__init__()

    def forward(self, inputs, targets):

        TP = (inputs * targets).sum()
        FP = (inputs * (1 - targets)).sum()

        return TP / (TP + FP)


class Recall(nn.Module):
    def __init__(self):
        super(Recall, self).__init__()

    def forward(self, inputs, targets):

        TP = (inputs * targets).sum()
        FN = (targets * (1-inputs)).sum()

        return TP / (TP + FN)

class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, inputs, targets):     

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        return intersection / union 


class DiceFocalLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceFocalLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        
    def forward(self, inp, tar):
        return self.dice(inp, tar) + self.focal(inp, tar)
