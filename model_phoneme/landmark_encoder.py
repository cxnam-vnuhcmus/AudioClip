
import torch
import torch.nn.functional as F
from torch import nn

#landmark encoder
#landmarks feature extractor
class LandmarkEncoder(nn.Module):
    def __init__(self, dim_out=128):
        super(LandmarkEncoder, self).__init__()
        self.dim_out = dim_out
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(40*2, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, dim_out),
            nn.LeakyReLU(0.02, True),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.encoder_fc1(x)
        return x