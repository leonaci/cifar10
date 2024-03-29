
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

"""
def get_dataloader(split, batch_size):
    import os
    import torch
    import torchvision.transforms as T
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

def get_dataloader(phase, batch_size):
    import os
    import torch
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    train = True if phase == 'train' else False
    shuffle = True if phase == 'train' else False
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    train_dataset = CIFAR10(root='../dataset', train=train, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True)
    
    return dataloader
