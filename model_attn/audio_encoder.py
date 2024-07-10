import torch
import torch.nn as nn
import torchvision.models as models

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        # Load pretrained ResNet-18
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children()))
        
    def forward(self, x):
        x = self.resnet18[0](x)
        x = self.resnet18[1](x)
        x = self.resnet18[2](x)
        x = self.resnet18[3](x)
        
        x1 = self.resnet18[4](x)
        x2 = self.resnet18[5](x1)
        x3 = self.resnet18[6](x2)
        x4 = self.resnet18[7](x3)
        
        return (x1, x2, x3, x4)
