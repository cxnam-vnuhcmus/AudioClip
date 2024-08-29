import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn

class ConvTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, kernel_size=3, dropout=0.1):
        super(ConvTransformerBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert (B, T, C) to (B, C, T) for Conv1d
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Convert back to (B, T, C)

        x = self.transformer_encoder(x)
        x = self.dropout(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, dim_in=80, d_model=80, num_heads=8, num_layers=4, dropout=0.1):
        super(AudioEncoder, self).__init__()
        self.conv_transformer = ConvTransformerBlock(d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        
        self.q_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.k_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.v_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.attention = ScaledDotProductAttention()

        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 128)

        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(128)

    def forward(self, x):
        # Pass through the Conv-Transformer block
        x = self.conv_transformer(x)  # Output shape: (B, T, d_model)

        # Apply 1x1 convolutions for q, k, v
        q = self.q_conv(x.permute(0, 2, 1))  # Shape: (B, d_model, T)
        k = self.k_conv(x.permute(0, 2, 1))  # Shape: (B, d_model, T)
        v = self.v_conv(x.permute(0, 2, 1))  # Shape: (B, d_model, T)

        # Transpose back to (B, T, d_model) for attention
        q = q.permute(0, 2, 1)  # Shape: (B, T, d_model)
        k = k.permute(0, 2, 1)  # Shape: (B, T, d_model)
        v = v.permute(0, 2, 1)  # Shape: (B, T, d_model)

        # Scaled dot-product attention
        attn_output, _ = self.attention(q, k, v)  # Shape: (B, T, d_model)

        # Apply final fully connected layers
        out = F.relu(self.fc1(attn_output))  # Shape: (B, T, 128)
        out = self.layer_norm1(out)          # Shape: (B, T, 128)
        out = self.fc2(out)                  # Shape: (B, T, 128)
        out = self.layer_norm2(out)          # Shape: (B, T, 128)

        out = out.mean(dim=1, keepdim=True)  # Shape: (B, 1, 128)
        return out