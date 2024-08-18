
import torch
import torch.nn as nn

class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(LandmarkEncoder, self).__init__()
        
        # Giả sử input_dim là 131*2 = 262
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x có shape (B, 5, 131, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        
        x = x.permute(0, 2, 1)  # Đổi shape thành (B, 262, 5) để phù hợp với Conv1d
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.permute(0, 2, 1)  # Đổi shape lại thành (B, 5, hidden_dim) để phù hợp với Transformer
                
        x = self.transformer_encoder(x)  # Xử lý bằng Transformer Encoder
        
        x = x[:, -1, :].unsqueeze(1)    # (B, 1, hidden_dim)
        
        x = self.fc(x)  # Chuyển đổi thành vector đầu ra
        
        return x