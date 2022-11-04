import torch
import torchvision
import torch.nn as nn
from .Encoder import resnet_maxpool_indices

def convT3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def convT1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def downsample(in_planes: int, out_planes: int, stride: int = 1):
    return [convT1x1(in_planes, out_planes, stride), nn.BatchNorm2d(out_planes)]

class RBottleneck(nn.Module):
    # ResNet V1.5 Bottleneck changed by replacing Conv2d with ConvTranspose2d
    # Origin:
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    def __init__(
        self,
        inp: int,
        inner: int,
        outp: int,
        stride: int = 1,
        dilation: int = 1,
        downsample = None,
    ):
        self.stride = stride
        super(RBottleneck, self).__init__()
        self.convT1 = convT1x1(inp, inner)
        self.bn1 = nn.BatchNorm2d(inner)
        self.convT2 = convT3x3(inner, inner, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(inner)
        self.convT3 = convT1x1(inner, outp)
        self.bn3 = nn.BatchNorm2d(outp)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        batch = x.shape[0]
        if self.stride == 1:
            output_size = x.shape
        else:
            output_size = [x.shape[2] * self.stride, x.shape[3] * self.stride]
        out = self.convT1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.convT2(out, output_size=output_size)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.convT3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample[0](x, output_size=output_size)
            identity = self.downsample[1](identity)
        out += identity
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(in_features=1000, out_features=2048, bias=True)
        # It's impossible to invert AdaptiveAvgPool2d but it may be simulated by ConvTranspose2d
        # https://discuss.pytorch.org/t/is-it-possible-to-find-the-inverse-of-adaptiveavgpool2d/124264
        self.avgpool = nn.ConvTranspose2d(2048, 2048, kernel_size=(8, 8), stride=(1, 1), bias=False)
        #
        self.layer4 = nn.Sequential(
            RBottleneck(2048, 512, 2048, stride=1),
            RBottleneck(2048, 512, 2048, stride=1),
            RBottleneck(2048, 512, 1024, stride=2, downsample=downsample(2048, 1024, 2))
        )
        self.layer3 = nn.Sequential(
            RBottleneck(1024, 256, 1024, stride=1),
            RBottleneck(1024, 256, 1024, stride=1),
            RBottleneck(1024, 256, 1024, stride=1),
            RBottleneck(1024, 256, 1024, stride=1),
            RBottleneck(1024, 256, 1024, stride=1),
            RBottleneck(1024, 256, 512, stride=2, downsample=downsample(1024, 512, 2))
        )
        self.layer2 = nn.Sequential(
            RBottleneck(512, 128, 512, stride=1),
            RBottleneck(512, 128, 512, stride=1),
            RBottleneck(512, 128, 512, stride=1),
            RBottleneck(512, 128, 256, stride=2, downsample=downsample(512, 256, 2))
        )
        self.layer1 = nn.Sequential(
            RBottleneck(256, 64, 256, stride=1),
            RBottleneck(256, 64, 256, stride=1),
            RBottleneck(256, 64, 256, stride=1),
            RBottleneck(256, 64, 64, stride=1, downsample=downsample(256, 64, 1)),
        )
        self.umaxpool_stride = 2
        self.umaxpool = nn.MaxUnpool2d(3, stride=self.umaxpool_stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1T = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
    def _forward_impl(self, x, resnet_maxpool_indices):
        x = self.fc(x)
        x = x.view(x.shape[0], 2048, 1, 1)
        x = self.avgpool(x)
        #
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        #
        x = self.umaxpool(x, resnet_maxpool_indices, output_size=[self.umaxpool_stride * x.shape[2], self.umaxpool_stride * x.shape[3]])
        x = self.relu(x)
        x = self.conv1T(x)
        x = self.bn1(x)
        return x
    def forward(self, x):
        global resnet_maxpool_indices
        return self._forward_impl(x, resnet_maxpool_indices)