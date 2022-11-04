import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Decoder(nn.Module):
    def __init__(self, dim=1000):
        super(Decoder, self).__init__()
        
        self.linear = nn.Sequential(
            self.linear_block(dim, 4096),
            self.linear_block(4096, 4096, dropout=True),
            self.linear_block(4096, 7*7*512, dropout=True),
        )

        self.main = nn.Sequential(
            self.maxpool_block(512, 512),
            self.conv_block(512),
            self.conv_block(512),
            self.conv_block(512),
            self.conv_block(512),

            self.maxpool_block(512, 512),
            self.conv_block(512),
            self.conv_block(512),
            self.conv_block(512),
            self.conv_block(512),
            
            self.maxpool_block(512, 256),
            self.conv_block(256),
            self.conv_block(256),
            self.conv_block(256),
            self.conv_block(256),

            self.maxpool_block(256, 128),
            self.conv_block(128),
            self.conv_block(128),

            self.maxpool_block(128, 64),
            self.conv_block(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True),
        )
        
    def conv_block(self, channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False)
        )
    def maxpool_block(self, input, output):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=input, out_channels=output, kernel_size=2, stride=2, padding=0, bias=True),
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
        x = self.linear(input)
        return self.main(x.view(input.shape[0], 512, 7, 7))