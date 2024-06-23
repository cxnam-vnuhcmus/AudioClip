import torch
import torch.nn as nn
import torchvision.models as models

class FaceImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(FaceImageEncoder, self).__init__()
        # Load a pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)
        # Remove the final fully connected layer
        # self.resnet = nn.Sequential(*list(self.resnet.children()))
        # Add a new fully connected layer for face encoding
        self.fc = nn.Linear(self.resnet.fc.out_features, 128)  # 128-dimensional face embeddings
        

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

