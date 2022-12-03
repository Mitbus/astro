import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class Encoder(nn.Module):
    def __init__(self, dim=1000, in_channels=1):
        # refereces for VGG19: https://arxiv.org/pdf/1409.1556v6.pdf
        super(Encoder, self).__init__()
        P = 0.5

        layer1 = self.create_layer(in_channels, 64)  # 128 -> 64
        layer2 = self.create_layer(64, 128)  # 64 -> 32
        layer3 = self.create_layer(128, 256, has_third=True)  # 32 -> 16
        layer4 = self.create_layer(256, 512, has_third=True)   # 16 -> 8
        layer5 = self.create_layer(512, 512, has_third=True)  # 8 -> 4

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
        )
        self.linear = nn.Sequential(
            self.linear_block(4*4*512, 4096),
            self.linear_block(4096, 4096),
            self.linear_block(4096, dim, activation=nn.Tanh())
        )

    def create_layer(self, base_channels_in, base_channels_out, has_third=False):
        if has_third:
            return nn.Sequential(
                self.conv_block(base_channels_in, base_channels_out),
                self.conv_block(base_channels_out, base_channels_out),
                self.maxpool_block(base_channels_out, base_channels_out)
            )
        else:
            return nn.Sequential(
                self.conv_block(base_channels_in, base_channels_out),
                self.maxpool_block(base_channels_out, base_channels_out)
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
