import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class Encoder(nn.Module):
    def __init__(self, dim=1000):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            self.conv_block(64),
            self.maxpool_block(64, 128),

            self.conv_block(128),
            self.conv_block(128),
            self.maxpool_block(128, 256),

            self.conv_block(256),
            self.conv_block(256),
            self.conv_block(256),
            self.conv_block(256),
            self.maxpool_block(256, 512),

            self.conv_block(512),
            self.conv_block(512),
            self.conv_block(512),
            self.conv_block(512),
            self.maxpool_block(512, 512),

            self.conv_block(512),
            self.conv_block(512),
            self.conv_block(512),
            self.conv_block(512),
            self.maxpool_block(512, 512),
        )
        self.linear = nn.Sequential(
            self.linear_block(7*7*512, 4096, dropout=True),
            self.linear_block(4096, 4096, dropout=True),
            self.linear_block(4096, dim, activation=nn.Tanh())
        )

    def conv_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
        )
    def maxpool_block(self, input, output):
        return nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=output, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=False),
        )
    def linear_block(self, input, output, activation=nn.ReLU(inplace=False), dropout=False):
        if dropout:
            return nn.Sequential(
                nn.Linear(in_features=input, out_features=output, bias=True),           
                activation,
                nn.Dropout(p=0.5, inplace=False),
            )
        else:
            return nn.Sequential(
                nn.Linear(in_features=input, out_features=output, bias=True),           
                activation,
            )
    
    def forward(self, input):
        x = self.main(input)
        return self.linear(x.flatten(1))
