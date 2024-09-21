import torch
import torch.nn as nn
import torch.nn.functional as F

class LLFEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128):
        super(LLFEncoder, self).__init__()
        
        # Lớp Conv1D để mã hóa input (B, N, 32) thành vector (B, 1, 128)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling để đưa về (B, 1, 128)

    def forward(self, x):
        # Input shape: (B, N, 32), chuyển về (B, 32, N) cho Conv1D
        x = x.permute(0, 2, 1)  # Đổi thứ tự trục thành (B, 32, N)
        
        # Conv1D layers
        x = F.relu(self.conv1(x))  # (B, 64, N)
        x = F.relu(self.conv2(x))  # (B, 128, N)
        
        # Pooling để đưa về (B, 128, 1)
        x = self.pool(x)  # (B, 128, 1)
        x = x.permute(0,2,1)
        
        return x