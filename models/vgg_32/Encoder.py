import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class Encoder(nn.Module):
    def __init__(self, dim=512):
        super(Encoder, self).__init__()
        P = 0.5

        base_channels = 32
        layer1 = nn.Sequential(
            self.conv_block(1, base_channels),
            self.conv_block(base_channels, base_channels),
            self.maxpool_block(base_channels, base_channels)
        )
        layer2 = nn.Sequential(
            self.conv_block(base_channels, base_channels*2),
            self.conv_block(base_channels*2, base_channels*2),
            self.maxpool_block(base_channels*2, base_channels*2)
        )

        base_channels *= 2
        layer3 = nn.Sequential(
            self.conv_block(base_channels, base_channels*2),
            self.conv_block(base_channels*2, base_channels*2),
            self.maxpool_block(base_channels*2, base_channels*2)
        )

        base_channels *= 2
        layer4 = nn.Sequential(
            self.conv_block(base_channels, base_channels*2),
            self.conv_block(base_channels*2, base_channels*2),
            self.maxpool_block(base_channels*2, base_channels*2)
        )

        base_channels *= 2
        layer5 = nn.Sequential(
            self.conv_block(base_channels, base_channels*2),
            self.conv_block(base_channels*2, base_channels*2),
            self.maxpool_block(base_channels*2, base_channels*2)
        )

        base_channels *= 2
        layer6 = nn.Sequential(
            self.conv_block(base_channels, base_channels*2),
            self.conv_block(base_channels*2, base_channels*2),
            self.maxpool_block(base_channels*2, base_channels*2)
        )
        base_channels *= 2
        layer7 = nn.Sequential(
            self.conv_block(base_channels, base_channels),
            self.conv_block(base_channels, base_channels),
            self.maxpool_block(base_channels, base_channels)
        )

        self.main = nn.Sequential(
            layer1,
            nn.Dropout(p=P, inplace=False),
            layer2,
            nn.Dropout(p=P, inplace=False),
            layer3,
            nn.Dropout(p=P, inplace=False),
            layer4,
            nn.Dropout(p=P, inplace=False),
            layer5,
            nn.Dropout(p=P, inplace=False),
            layer6,
            nn.Dropout(p=P, inplace=False),
            layer7
        )
        base_channels = 256
        self.linear = nn.Sequential(
            self.linear_block(4*base_channels, 4*base_channels),
            self.linear_block(4*base_channels, 4*base_channels),
            self.linear_block(4*base_channels, 2*base_channels),
            self.linear_block(2*base_channels, dim, activation=nn.Tanh())
        )

    def conv_block(self, input, output):
        return nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(in_channels=output, out_channels=output, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output),
        )
    def maxpool_block(self, input, output):
        return nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.1, inplace=False)
        )
    def linear_block(self, input, output, activation=nn.ReLU(inplace=False)):
        return nn.Sequential(
            nn.Linear(in_features=input, out_features=output, bias=True),           
            activation
        )
    
    
    def forward(self, input):
        x = self.main(input).flatten(1)
        return self.linear(x)
