import torch
import torchvision
import torch.nn as nn

resnet_maxpool_indices = None

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # loading pretrained nvidia_resnet50
        self.resnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False, return_indices=True)
        self.resnet.fc = nn.Linear(in_features=1024, out_features=1000, bias=True)
    
    def _forward_impl(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x, resnet_maxpool_indices = self.resnet50.maxpool(x)
        x = self.resnet.layers[0](x)
        x = self.resnet.layers[1](x)
        x = self.resnet.layers[2](x)
        x = self.resnet.avgpool(x)
        x = x.flatten(1)
        x = self.resnet.fc(x)
        return x, resnet_maxpool_indices

    def forward(self, x):
        global resnet_maxpool_indices
        x, resnet_maxpool_indices = self._forward_impl(x)
        return x
