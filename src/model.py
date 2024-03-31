import torch
import torch.nn as nn
import torch.nn.functional as F

## (batch_size, channels, w, h) -> (batch_size, channels, w, h)
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

        ## (batch_size, channels, w, h) -> (batch_size, channels, w // 2, h // 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        ## (batch_size, 3, 32, 32) -> (batch_size, 64, 32, 32)
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        ## (batch_size, 64, 16, 16) -> (batch_size, 128, 8, 8)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        ## (batch_size, 128, 8, 8) -> (batch_size, 256, 4, 4)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        ## (batch_size, 256, 4, 4) -> (batch_size, 512, 2, 2)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        ## (batch_size, 512, w, h) -> (batch_size, 512, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        ## (batch_size, 512) -> (batch_size, 10)
        self.classifier = nn.Linear(512, 10)
        
        self.to(device)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
