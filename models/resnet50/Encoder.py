import torch
import torchvision
import torch.nn as nn

resnet_maxpool_indices = None

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # loading pretrained nvidia_resnet50
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        # 3x256x256 -> 1x256x256
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # return_indices for decoder
        self.resnet50.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, return_indices=True)
    
    def _forward_impl(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x, resnet_maxpool_indices = self.resnet50.maxpool(x)
        x = self.resnet50.layers(x)
        x = self.resnet50.avgpool(x)
        x = x.flatten(1)
        x = self.resnet50.fc(x)
        return x, resnet_maxpool_indices

    def forward(self, x):
        global resnet_maxpool_indices
        x, resnet_maxpool_indices = self._forward_impl(x)
        return x
