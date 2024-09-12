import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FACEMESH_ROI_IDX, FACEMESH_LIPS_IDX, FACEMESH_FACES_IDX

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        # Tính điểm attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn

class LandmarkDecoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=131*2):
        super(LandmarkDecoder, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim )
        self.shortcut1 = nn.Linear(input_dim, hidden_dim)
        
        self.conv1d = nn.Conv1d(in_channels=input_dim + 32, out_channels=hidden_dim, kernel_size=1)  # Conv1D to match dimensions
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        
        self.linear3 = nn.Linear(hidden_dim, output_dim )
        

    def forward(self, audio_feature, llfs_feature, lm_feature):
        combined_feature = audio_feature + lm_feature  # (B, N, 128)
        
        # Thực hiện các lớp Linear
        x = self.linear1(combined_feature)  # (B, N, hidden_dim)
        x = F.relu(x)  # Áp dụng ReLU activation function
        x = self.linear2(x)  # (B, N, output_dim)
        shortcut1 = self.shortcut1(combined_feature)  # (B, N, output_dim)
        x = x + shortcut1  # (B, N, output_dim)
        
        y = torch.cat((audio_feature,llfs_feature), dim=-1)
        y = self.conv1d(y.transpose(1, 2)).transpose(1, 2)  # Shape: (B, 1, 128)
        y = self.layer_norm(y)  # Normalized k, v
        
        output, _ = self.mha(x, y, y)  # (q, k, v)
        
        output = self.linear3(output)
        
        #reshape
        output = output.reshape(output.shape[0],output.shape[1],-1,2)
        
        return output
