import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        outputs = self.relu(x)
        return outputs
    
class SpectraEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, input_length=4000):
        super().__init__()
        self.s1 = nn.Sequential(
            Block(in_channels, 32),
            nn.Conv1d(32, 32, kernel_size=3, stride=2),
        )

        self.s2 = nn.Sequential(
            Block(32, 64),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),
        )

        self.s3 = nn.Sequential(
            Block(64, 128), 
            nn.Conv1d(128, 128, kernel_size=3, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self._get_flat_size(in_channels, input_length), latent_dim)

    def _get_flat_size(self, in_channels, input_length):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            x = self.s1(dummy)
            x = self.s2(x)
            x = self.s3(x)
            x = self.pool(x)
            x = self.flatten(x)
            return x.shape[1]

    def forward(self, inputs):
        x = self.s1(inputs)
        x = self.s2(x)
        x = self.s3(x)

        x = self.pool(x)
        x = self.flatten(x)
        outputs = self.linear(x)
        return outputs


class Down(nn.Module):
    def  __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    
    def forward(self, x):
        conv = self.conv(x)
        outputs = self.pool(conv)
        return outputs, conv

class Up(nn.Module):
    def  __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, inputs, layerc):
        x = self.up(inputs)
        x = torch.concat([layerc, x], dim=1)
        outputs = self.conv(x)
        return outputs

class UNet(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, spectra_length=4000):
        super().__init__()
        self.downs = nn.ModuleList([
            Down(in_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
        ])

        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=1), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.ups = nn.ModuleList([
            Up(1024, 512),
            Up(512, 256),
            Up(256, 128),
            Up(128, 64),
        ])

        self.outConv = nn.Conv1d(64, 2, kernel_size=1)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(spectra_length * 2, latent_dim)

    def forward(self, x):
        outs = []
        for block in self.downs:
            x, layerc = block(x)
            outs.append(layerc)

        x = self.bottleneck(x)

        for block in self.ups:
            layerc = outs.pop()
            x = block(x, layerc)
        
        x = self.outConv(x)
        x = self.flatten(x)
        outputs = self.head(x)
        return outputs 
        