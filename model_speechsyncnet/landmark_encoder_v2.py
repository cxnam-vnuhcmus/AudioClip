import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling 1D
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.Sigmoid()
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        B, N, C = x.shape # (B, N, 256)        
        x = x.permute(0,2,1) # (B, 256, N)
        
        x1 = self.gap(x)     # (B, 256, 1)
        x1 = x1.permute(0,2,1)# (B, 1, 256)
        x1 = self.fc(x1) # (B, 1, 128)
        
        x2 = self.conv(x) # (B, 128, N)
        x2 = x2.permute(0,2,1) # (B, N, 128)

        out = x1 * x2 # (B, N, 128)
        return out

class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim=(131, 2), output_dim=(5,128), hidden_size=256):
        super(LandmarkEncoder, self).__init__()
        input_size = input_dim[0] * input_dim[1]
        output_size = output_dim[0] * output_dim[1]
        
        self.fc_bn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size)  
        )

        # Channel Attention Module
        self.channel_attention = ChannelAttention(input_dim=hidden_size, output_dim=128)
        

    def forward(self, x):
        B, N, C1, C2 = x.shape  # (B, N, 131, 2)        
        x = x.reshape(B, N, -1)   # (B,N, 262)        
        x = self.fc_bn(x)               # (B, N, 256)

        x = self.channel_attention(x)  # Kết quả shape vẫn là (B, N, 128)
        
        return x
