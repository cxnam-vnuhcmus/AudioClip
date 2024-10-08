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
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=128, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)
        
        

    def forward(self, x):
        B, N, C = x.shape # (B, N, 256)
        x1 = x.permute(0,2,1) # (B, 256, N)
        
        avg_pool = self.gap(x1).squeeze(-1)  # (B, 256, 1) -> (B, 256)
        fc_output = self.fc(avg_pool).unsqueeze(1) # (B, 1, 256)
        
        conv_output = self.conv(x1) # (B, 256, N)
        conv_output = conv_output.permute(0,2,1) # (B, N, 256)

        out = fc_output * conv_output # (B, N, 256)
        
        out, _ = self.lstm(out) #(B, N, 128)
    
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

        self.fc_trans = nn.Linear(in_channels, hidden_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
        )
        
        self.fc = nn.Linear(in_channels, output_size)

        self.lstm = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)

        self.fc_out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        B, N, C1, C2 = x.shape  # (B, N, 131, 2)
        
        x = x.reshape(B, N, -1)  # (B, N, 262)
        
        x1 = x.permute(0,2,1)   # (B, 262, N)        
        x1 = self.conv(x1)      # (B, 256, N)
        x1 = x1.permute(0,2,1)   # (B, N, 256)
        
        x = self.fc_trans(x)       # (B, N, 256) 
        x = self.transformer_encoder(x) # (B, N, 256)
        
        x = torch.cat((x,x1), dim = -1)
        
        x, _ = self.lstm(x) # (B, N, 512)
        
        x = x[:, -1, :] #(B, 512)
        
        x = self.fc_out(x).unsqueeze(1) #(B, 1, 128)
        
        return x
