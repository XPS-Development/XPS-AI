import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, dilation=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels, 
                kernel_size, 
                stride,
                padding, 
                dilation
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    

class XPSModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.starter = nn.Sequential(
            Block(1, 16),
            Block(16, 20),
            nn.AvgPool1d(kernel_size=2),
            Block(20, 24),
            nn.AvgPool1d(kernel_size=2),
            Block(24, 28),
            nn.AvgPool1d(kernel_size=2),
            Block(28, 32)
        )

        self.pass_down1 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            Block(32, 48)
        )

        self.pass_down2 = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            Block(48, 64)
        )

        self.code = nn.Sequential(
            nn.AvgPool1d(kernel_size=2),
            Block(64, 96),
            nn.Upsample(scale_factor=2),
            Block(96, 64)
        )

        self.pass_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(128, 64)
        )

        self.pass_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(112, 48)
        )

        self.finisher = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Block(80, 64),
            nn.Upsample(scale_factor=2),
            Block(64, 32),
            nn.Upsample(scale_factor=2),
            Block(32, 16),
            nn.Conv1d(16, 2, 1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        starter = self.starter(x)
        pass1 = self.pass_down1(starter)
        pass2 = self.pass_down2(pass1)
        x = self.code(pass2)
        x = torch.cat((x, pass2), dim=1)
        x = self.pass_up2(x)
        x = torch.cat((x, pass1), dim=1)
        x = self.pass_up1(x)
        x = torch.cat((x, starter), dim=1)
        x = self.finisher(x)
        peak_mask = x[:, 0, :]
        max_mask = x[:, 1, :]
        peak_mask = self.sigmoid(peak_mask)
        max_mask = self.sigmoid(max_mask)
        return peak_mask, max_mask
