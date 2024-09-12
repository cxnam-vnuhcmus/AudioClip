import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkDecoder(nn.Module):
    def __init__(self, input_dim=128*2, hidden_dim=128, output_dim=131*2):
        super(LandmarkDecoder, self).__init__()
        self.output_dim = output_dim
        
        # Lớp tuyến tính để chuyển đổi từ input_dim sang hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        
        # Lớp tuyến tính để chuyển đổi từ hidden_dim sang output_dim
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        # Lớp tuyến tính để chuyển đổi từ input_dim sang output_dim cho kết nối tắt
        self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, audio_feature, landmark_feature):
        # Kiểm tra shape đầu vào
        assert audio_feature.shape == landmark_feature.shape, "audio_feature and landmark_feature must have the same shape"
        
        # Kết hợp audio_feature và landmark_feature
        # combined_feature = audio_feature + landmark_feature  # (B, N, 128)
        combined_feature = torch.cat((audio_feature, landmark_feature), dim=-1) # (B, N, 128*2)
        
        # Thực hiện các lớp Linear
        x = self.linear1(combined_feature)  # (B, N, hidden_dim)
        x = F.relu(x)  # Áp dụng ReLU activation function
        x = self.linear2(x)  # (B, N, output_dim)
        
        # Kết nối tắt
        shortcut = self.shortcut(combined_feature)  # (B, N, output_dim)
        
        # Kết hợp đầu ra của lớp Linear với kết nối tắt
        output = x + shortcut  # (B, N, output_dim)
        
        #reshape
        output = output.reshape(output.shape[0],output.shape[1],-1,2)
        
        return output
