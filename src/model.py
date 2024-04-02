import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
    def __init__(self, depth=2):
        super(ImageClassifier, self).__init__()

    def forward(self, x):
        return x
