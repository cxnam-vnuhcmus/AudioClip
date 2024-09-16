import torch
import torch.nn as nn
import torch.nn.functional as F

# LLFEncoder: Module dùng để xử lý LLFs
class LLFEncoder(nn.Module):
    def __init__(self, input_dim=32, output_dim=128):
        super(LLFEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=output_dim, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Thêm global average pooling để giảm chiều dài chuỗi xuống 1
        
    def forward(self, x):
        # x shape: (B, 5, 32) -> cần đổi shape để Conv1d làm việc
        x = x.permute(0, 2, 1)  # Đổi thành (B, 32, 5)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.global_pool(x)  # Áp dụng pooling để giảm (B, 128, 5) -> (B, 128, 1)
        x = x.permute(0, 2, 1)  # Đổi lại thành (B, 1, 128)
        return x
