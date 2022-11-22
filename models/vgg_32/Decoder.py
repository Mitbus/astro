import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Decoder(nn.Module):
    def __init__(self, dim=512):
        super(Decoder, self).__init__()
        P = 0.5
        
        self.linear = nn.Sequential(
            self.linear_block(dim, 1024),
            self.linear_block(1024, 2048),
            self.linear_block(2048, 2048),
            self.linear_block(2048, 2048)
        )

        base_channel = 128
        layer1 = nn.Sequential(
            self.rconv_block(base_channel, base_channel // 2),
            self.rconv_block(base_channel // 2, base_channel // 2),
            self.rmaxpool_block(base_channel // 2, base_channel // 2)
        )
        base_channel = base_channel // 2
        layer2 = nn.Sequential(
            self.rconv_block(base_channel, base_channel // 2),
            self.rconv_block(base_channel // 2, base_channel // 2),
            self.rmaxpool_block(base_channel // 2, base_channel // 2)
        )
        base_channel = base_channel // 2
        layer3 = nn.Sequential(
            self.rconv_block(base_channel, base_channel // 2),
            self.rconv_block(base_channel // 2, base_channel // 2),
            self.rmaxpool_block(base_channel // 2, base_channel // 2)
        )
        base_channel = base_channel // 2
        layer4 = nn.Sequential(
            self.rconv_block(base_channel, base_channel // 2),
            self.rconv_block(base_channel // 2, base_channel // 2),
            self.rmaxpool_block(base_channel // 2, base_channel // 2)
        )
        base_channel = base_channel // 2
        layer5 = nn.Sequential(
            self.rconv_block(base_channel, base_channel // 2),
            self.rconv_block(base_channel // 2, base_channel // 2),
            self.rmaxpool_block(base_channel // 2, base_channel // 2)
        )
        base_channel = base_channel // 2
        layer6 = nn.Sequential(
            self.rconv_block(base_channel, base_channel // 2),
            self.rconv_block(base_channel // 2, 1),
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
            layer6
        )
        
    def rconv_block(self, input, output):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=input, out_channels=output, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.1, inplace=False),
            nn.ConvTranspose2d(in_channels=output, out_channels=output, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output),
        )
    def rmaxpool_block(self, input, output):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=input, out_channels=output, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.1, inplace=False)
        )
    def linear_block(self, input, output, activation=nn.ReLU(inplace=False)):
        return nn.Sequential(
            nn.Linear(in_features=input, out_features=output, bias=True),           
            activation
        )

    def forward(self, input):
        x = self.linear(input).view(input.shape[0], 128, 4, 4)
        return self.main(x)