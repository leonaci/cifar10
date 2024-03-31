import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNetBlock

class ImageClassifier(nn.Module):
    def __init__(self, depth=2):
        super(ImageClassifier, self).__init__()

        self.depth = depth

        ## (batch_size, channels, w, h) -> (batch_size, channels, w // 2, h // 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        ## (batch_size, channels, w, h) -> (batch_size, channels, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        ## (batch_size, 3, 32, 32) -> (batch_size, 64, 16, 16)
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        ## (batch_size, 64, 16, 16) -> (batch_size, 128, 8, 8)
        self.layer1 = ResNetBlock(in_channels=64, out_channels=128, stride=2)

        layers = []
        for _ in range(self.depth):
            layers.append(ResNetBlock(in_channels=128, out_channels=128))
        
        ## (batch_size, 128, 8, 8) -> (batch_size, 128, 8, 8)
        self.conv1 = nn.Sequential(*layers)

        ## (batch_size, 128, 8, 8) -> (batch_size, 256, 4, 4)
        self.layer2 = ResNetBlock(in_channels=128, out_channels=256, stride=2)

        layers = []
        for _ in range(self.depth):
            layers.append(ResNetBlock(in_channels=256, out_channels=256))
        
        ## (batch_size, 256, 4, 4) -> (batch_size, 256, 4, 4)
        self.conv2 = nn.Sequential(*layers)

        ## (batch_size, 256, 4, 4) -> (batch_size, 512, 2, 2)
        self.layer3 = ResNetBlock(in_channels=256, out_channels=512, stride=2)

        layers = []
        for _ in range(self.depth):
            layers.append(ResNetBlock(in_channels=512, out_channels=512))

        ## (batch_size, 512, 2, 2) -> (batch_size, 512, 2, 2)
        self.conv3 = nn.Sequential(*layers)

        ## (batch_size, 512) -> (batch_size, 10)
        self.classifier = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, ResNetBlock):
               nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.input_layer(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.avg_pool(x).flatten(1)

        x = self.classifier(x)
        return x
