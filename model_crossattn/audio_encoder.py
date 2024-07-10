import torch
import torch.nn as nn
import torchvision.models as models

class AudioEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(AudioEncoder, self).__init__()
        # Load pretrained ResNet-18
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children()))
        
        # with torch.no_grad():
        #     dummy_input = torch.randn(1, 3, 224, 224)
        #     dummy_output = self.resnet18(dummy_input)
        #     resnet_output_dim = dummy_output.view(dummy_output.size(0), -1).size(1)
        
        # self.fc = nn.Sequential(
        #     nn.Linear(resnet_output_dim, 3*64*64),
        #     nn.ReLU(),
        #     nn.Linear(3*64*64, output_dim)
        # )

    def forward(self, x):
        # x = self.resnet18(x)
        x = self.resnet18[0](x)
        x = self.resnet18[1](x)
        x = self.resnet18[2](x)
        x = self.resnet18[3](x)
        
        x1 = self.resnet18[4](x)
        x2 = self.resnet18[5](x1)
        x3 = self.resnet18[6](x2)
        x4 = self.resnet18[7](x3)
        
        # x = self.resnet18.avgpool(x4)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return (x1, x2, x3, x4)
