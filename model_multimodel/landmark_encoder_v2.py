import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        # Nhánh 1: Global Average Pooling + Fully Connected + Sigmoid
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling 1D
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        # Nhánh 2: Convolution + ReLU + Convolution + Sigmoid
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C = x.shape
        x1 = x.permute(0,2,1)
        
        # Nhánh 1: Global Average Pooling + Fully Connected + Sigmoid
        avg_pool = self.gap(x1).squeeze(-1)  # GAP với đầu vào (B*N, 262, 1) → đầu ra (B*N, 262)
        fc_output = self.fc(avg_pool).unsqueeze(-1) # Thêm chiều cuối   (B*N, 128, 1)
        
        # Nhánh 2: Convolution + ReLU + Convolution + Sigmoid
        conv_output = self.conv(x) # (B*N, 128, 1)
        conv_output = conv_output.permute(0,2,1)

        # Nhân kết quả của 2 nhánh
        out = fc_output * conv_output
        
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
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, dropout=dropout, bidirectional=False)
        

        # Channel Attention Module
        self.channel_attention = ChannelAttention(in_channels=hidden_size, reduction=reduction)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, x):
        B, N, C1, C2 = x.shape  # (B, N, 131, 2)
        
        x = x.reshape(B*N, -1)   # (B,N, 262)
        
        x = self.fc_bn(x)               # (B,N, 256)
                
        x = x.reshape(B, N, -1)         # (B, N, 256)
        
        x = self.transformer_encoder(x) # (B, N, 256)
        
        x1, _ = self.lstm(x)
        
        x = x1 + x
        
        x = x[:, -1, :].unsqueeze(1)    # (B, 1, 256)
        
        # Channel Attention
        x = self.channel_attention(x)  # Kết quả shape vẫn là (B, 256, 1)
        x = x.squeeze(-1)  # Bỏ chiều cuối: (B, 256)

        # Fully Connected Layers
        out = self.fc(x).unsqueeze(1)  # Kết quả shape là (B, 128)
        
        return out
