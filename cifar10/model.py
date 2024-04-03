import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
    def __init__(self, config):
        super(ImageClassifier, self).__init__()

    def forward(self, x):
        out = x

        return out
