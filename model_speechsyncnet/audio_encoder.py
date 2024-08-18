import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerWithAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout):
        super(TransformerWithAttention, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Đưa x vào Transformer Encoder
        x = self.transformer_encoder(x)
        x = x[:,-1,:].unsqueeze(1)
        
        return x

class AudioEncoder(nn.Module):
    def __init__(self, dim_in=80, dim_out=128):
        super(AudioEncoder, self).__init__()
        d_model = dim_in  # Kích thước embedding của Transformer Encoder
        num_heads = 8
        num_layers = 4
        dropout = 0.1

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, dim_out)

        
    def forward(self, x):        
        # Tiến hành forward pass
        x = self.transformer_encoder(x)
        
        x = x[:,-1,:].unsqueeze(1)
        
        # Kết quả cuối cùng
        output = self.fc(x)
        return output