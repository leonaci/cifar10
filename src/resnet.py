from typing import Any, Callable, List, Optional, Type, Union

import torch
from torch import Tensor 
import torch.nn as nn
import torch.nn.functional as F

## (batch_size, in_channels, w, h) -> (batch_size, out_channels, (w - 1) // stride + 1, (h - 1) // stride + 1) 
def conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd!")
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=stride,
        bias=False
    )

def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    return conv(in_channels, out_channels, kernel_size=1, stride=stride)

def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    return conv(in_channels, out_channels, kernel_size=3, stride=stride)

def conv5x5(in_channels: int, out_channels: int, stride: int = 1):
    return conv(in_channels, out_channels, kernel_size=5, stride=stride)

def conv7x7(in_channels: int, out_channels: int, stride: int = 1):
    return conv(in_channels, out_channels, kernel_size=7, stride=stride)

## (batch_size, in_channels, w, h) -> (batch_size, out_channels, (w - 1) // stride + 1, (h - 1) // stride + 1)
class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, short: bool = True):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )

        self.short = short


    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.short:
            out += self.shortcut(x)

        out = self.relu(out)

        return out

## (batch_size, in_channels, w, h) -> (batch_size, out_channels, (w - 1) // stride + 1, (h - 1) // stride + 1)
class ResNetBottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, short: bool = True):
        super().__init__()

        self.conv1 = conv1x1(in_channels, out_channels//4)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels//4, out_channels//4, stride)
        self.bn2 = nn.BatchNorm2d(out_channels//4)

        self.conv3 = conv1x1(out_channels//4, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )

        self.short = short


    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.short:
            out += self.shortcut(x)

        out = self.relu(out)

        return out
