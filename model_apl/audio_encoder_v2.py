import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, N, d_model)
        N = x.size(1)
        return x + self.pe[:, :N, :].to(x.device)

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
        return output


class TransformerWithAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout):
        super(TransformerWithAttention, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.k_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.v_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.attention = ScaledDotProductAttention()

    def forward(self, x):
        x = self.transformer_encoder(x)
        
        q = x
        k = self.k_conv(x.permute(0, 2, 1))  # (batch_size, d_model, num_frames)
        v = self.v_conv(x.permute(0, 2, 1))  # (batch_size, d_model, num_frames)

        k = k.permute(0, 2, 1)  # (batch_size, num_frames, d_model)
        v = v.permute(0, 2, 1)  # (batch_size, num_frames, d_model)

        output = self.attention(q, k, v)
        return output

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, nhead=8, num_layers=4, dropout=0.1):
        super(AudioEncoder, self).__init__()
        
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
                
        self.transformer_encoder = TransformerWithAttention(d_model=hidden_dim, num_heads=nhead, num_layers=num_layers, dropout=dropout)
        
    def forward(self, x):       
        x = self.fc_in(x)  # (B, N, 128)
        x = self.pos_encoder(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # (B, N, 128)
        
        return x