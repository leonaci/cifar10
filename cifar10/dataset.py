import os
from . import PROJECT_ROOT
import torch
from torch.utils.data import Dataset

class CIFAR10(Dataset):
    def __init__(self, transform=None, split="train"):
        from datasets import load_dataset
        self.dataset = load_dataset("cifar10", split=split, cache_dir=os.path.join(PROJECT_ROOT, "dataset"))
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = self.dataset["img"][idx]
        label = self.dataset["label"][idx]
        img = img if self.transform is None else self.transform(img)
        label = torch.tensor(label)
        return img, label
