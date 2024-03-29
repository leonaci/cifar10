import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

"""
def get_dataloader(split, batch_size):
    from dataset import CIFAR10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    shuffle = True if phase == 'train' else False
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        T.Lambda(lambda x: x.to(device))
    ])
    
    train_dataset = CIFAR10(transform=transform, split=split)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True)
    
    return dataloader
"""

train_transform = T.Compose([
    T.RandomAffine(0, scale=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation([90, 90]),
    T.RandomRotation([180, 180]),
    T.RandomRotation([270, 270]),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

valid_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def get_dataloader(phase, batch_size):
    from torchvision.datasets import CIFAR10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    train = True if phase == 'train' else False
    shuffle = True if phase == 'train' else False
    transform = train_transform if phase == 'train' else valid_transform
    
    train_dataset = CIFAR10(root='../dataset', train=train, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True)
    
    return dataloader
