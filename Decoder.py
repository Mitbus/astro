import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Decoder(nn.Module):
    def __init__(self, dim=1000, out_channels=1):
        super(Decoder, self).__init__()
        P = 0.5
        
        self.linear = nn.Sequential(
            self.linear_block(dim, 4096),
            self.linear_block(4096, 4096),
            self.linear_block(4096, 4*4*512)
        )

        layer1 = self.create_layer(512, 512, has_third=True)  # 4 -> 8
        layer2 = self.create_layer(512, 256, has_third=True)  # 8 -> 16
        layer3 = self.create_layer(256, 128, has_third=True)  # 16 -> 32
        layer4 = self.create_layer(128, 64)  # 32 -> 64
        layer5 = self.create_layer(64, out_channels)  # 64 -> 128

        self.main = nn.Sequential(
            layer1,
            nn.Dropout(p=P, inplace=False),
            layer2,
            nn.Dropout(p=P, inplace=False),
            layer3,
            nn.Dropout(p=P, inplace=False),
            layer4,
            nn.Dropout(p=P, inplace=False),
            layer5
        )

    def create_layer(self, base_channels_in, base_channels_out, has_third=False):
        if has_third:
            return nn.Sequential(
                self.rmaxpool_block(base_channels_in, base_channels_in),
                self.rconv_block(base_channels_in, base_channels_in),
                self.rconv_block(base_channels_in, base_channels_out)
            )
        else:
            return nn.Sequential(
                self.rmaxpool_block(base_channels_in, base_channels_in),
                self.rconv_block(base_channels_in, base_channels_out)
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
        x = self.linear(input).view(input.shape[0], 512, 4, 4)
        return self.main(x)