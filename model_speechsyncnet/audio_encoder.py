import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TransformerWithAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout):
        super(TransformerWithAttention, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.q_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.k_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.v_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.attention = ScaledDotProductAttention()

    def forward(self, x):
        # Đưa x vào Transformer Encoder
        x = self.transformer_encoder(x)
        x = x[:,-1,:].unsqueeze(1)
        
        # Chia thành các nhánh q, k, v
        # q = self.q_conv(x.permute(0, 2, 1))  # (batch_size, d_model, num_frames)
        q = x
        k = self.k_conv(x.permute(0, 2, 1))  # (batch_size, d_model, num_frames)
        v = self.v_conv(x.permute(0, 2, 1))  # (batch_size, d_model, num_frames)

        # q = q.permute(0, 2, 1)  # (batch_size, num_frames, d_model)
        k = k.permute(0, 2, 1)  # (batch_size, num_frames, d_model)
        v = v.permute(0, 2, 1)  # (batch_size, num_frames, d_model)

        # Thực hiện scaled dot-product attention
        output, attn = self.attention(q, k, v)
        return output, attn

class AudioEncoder(nn.Module):
    def __init__(self, dim_in=80):
        super(AudioEncoder, self).__init__()
        d_model = dim_in  # Kích thước embedding của Transformer Encoder
        num_heads = 8
        num_layers = 4
        dropout = 0.1
        hidden_size1 = 128  # Kích thước hidden state của LSTM1
        hidden_size2 = 128  # Kích thước hidden state của LSTM2
        
        self.model = TransformerWithAttention(d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        
        self.lstm1 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size1, num_layers=1, batch_first=True, dropout=dropout, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, num_layers=1, batch_first=True, dropout=dropout, bidirectional=False)
        
        # Lớp kết nối tắt
        self.fc_attention = nn.Linear(d_model, hidden_size1)  # Đưa attention_output về kích thước hidden_size1
        self.fc = nn.Linear(hidden_size2, 128)

        
    def forward(self, x):        
        # Tiến hành forward pass
        attention_output, attn = self.model(x)
        
        # LSTM1 với kết nối tắt
        attention_output_proj = self.fc_attention(attention_output)  # (batch_size, num_frames, hidden_size1)
        lstm1_output, _ = self.lstm1(attention_output_proj)
        lstm1_output = lstm1_output + attention_output_proj  # Kết nối tắt

        # LSTM2 với kết nối tắt
        lstm2_output, _ = self.lstm2(lstm1_output)
        lstm2_output = lstm2_output + lstm1_output  # Kết nối tắt

        # Kết quả cuối cùng
        output = self.fc(attention_output_proj)
        return output
