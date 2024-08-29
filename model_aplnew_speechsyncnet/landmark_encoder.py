import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, input_dim=256):
        super(ChannelAttention, self).__init__()
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers to process the pooled features
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # Conv1D pathway for retaining spatial information
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, input_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(input_dim // 2, input_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, N, 256)
        
        # Global Average Pooling pathway
        avg_pooled = self.global_avg_pool(x.permute(0, 2, 1))  # (B, 256, 1)
        avg_pooled = avg_pooled.view(avg_pooled.size(0), -1)   # (B, 256)
        avg_pooled = self.fc(avg_pooled)                       # (B, 256)
        
        # Conv1D pathway
        conv_out = self.conv1d(x.permute(0, 2, 1))  # (B, 256, N)
        conv_out = conv_out.mean(dim=-1)            # (B, 256)
        
        # Combining both pathways
        out = avg_pooled * conv_out  # Element-wise multiplication to combine features
        
        # Expanding dimension to match the desired output shape (B, 1, 256)
        out = out.unsqueeze(1)  # (B, 1, 256)
        
        return out

class LandmarkEncoder(nn.Module):
    def __init__(self, input_size=(131, 2), output_size=128, hidden_size=256, reduction=16):
        super(LandmarkEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        in_channels = input_size[0] * input_size[1]  # 131 * 2 = 262        
        num_heads = 8
        num_layers = 4
        dropout = 0.1

        self.fc_bn = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(num_features=hidden_size)  
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Channel Attention Module
        self.channel_attention = ChannelAttention(input_dim=hidden_size)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, x):
        B, N, C1, C2 = x.shape  # (B, N, 131, 2)
        
        x = x.reshape(B * N, -1)   # (B*N, 262)
        
        x = self.fc_bn(x)               # (B*N, 256)
        
        x = x.reshape(B, N, -1)         # (B, N, 256)
        
        x = self.transformer_encoder(x) # (B, N, 256)
        
        # x = x[:, -1, :].unsqueeze(2)    # (B, 256, 1)
        # x = x.transpose(1, 2)  # (B, 256, N)
         
        # Channel Attention
        x = self.channel_attention(x)  # Kết quả shape vẫn là (B, 256, 1)
        x = x.squeeze(1)  # Bỏ chiều cuối: (B, 256)

        # Fully Connected Layers
        out = self.fc(x).unsqueeze(1)  # Kết quả shape là (B, 128)

        return out
