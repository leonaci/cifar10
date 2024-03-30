import torch
import torch.nn as nn
import torch.nn.functional as F

## (1, channels, w, h) -> (1, channels, w, h)
def conv(channels, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd!")

    return nn.Sequential(
        nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True)
    )

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        
        ## (1, 3, 32, 32) -> (1, 64, 16, 16)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        ## (1, 64, 16, 16) -> (1, 256, 8, 8)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        ## (1, 256, 8, 8) -> (1, 128, 4, 4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        ## (1, 128, w, h) -> (1, 128, w, h)
        self.conv1 = conv(128)

        ## (1, 128 * 4 * 4) -> (1, 10)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )
        
        self.to(device)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.conv1(x) + x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
