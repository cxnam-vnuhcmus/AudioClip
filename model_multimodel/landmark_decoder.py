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

class BottleneckBlock1D(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, residual=True):
        super(BottleneckBlock1D, self).__init__()
        
        # 1x1 Conv1d (reduce dimensions)
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)
        
        # 3x3 Conv1d (spatial conv, since we only have 2 spatial dims)
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        
        # 1x1 Conv1d (restore dimensions)
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # Option to add residual connection
        self.residual = residual
        if self.residual and in_channels != out_channels:
            # 1x1 Conv1d to match the output dimensions for residual connection
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.residual_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        B, N, C, D = x.shape
        x = x.view(B*N, C, D)

        # Save the input for residual connection
        identity = x

        # Reshape input to (B, C, N*2) for Conv1d
        x = x.transpose(1, 2)  # Shape becomes (B, C, N*D)
        

        # First 1x1 Conv1d
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 3x3 Conv1d
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Second 1x1 Conv1d
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Residual connection (if enabled)
        if self.residual:
            identity = identity.transpose(1, 2)
            if hasattr(self, 'residual_conv'):
                identity = self.residual_conv(identity)
                identity = self.residual_bn(identity)
            out += identity

        # Final activation function
        out = F.relu(out)

        # Reshape output back to (B, N, 131, 2)
        out = out.transpose(1, 2).unsqueeze(1)
        
        return out
    
class KANLayer(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=128):
        super(KANLayer, self).__init__()
        # Linear layers for the 1D transformations (Kolmogorov-Arnold decomposition)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Apply nonlinear transformation
        x = F.relu(self.fc1(x))  # First transformation
        x = F.relu(self.fc2(x))  # Second transformation
        x = self.fc3(x)          # Final output
        return x

class KAN(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=128):
        super(KAN, self).__init__()
        # A KAN layer for each landmark point
        self.kan_layer = KANLayer(input_dim, output_dim, hidden_dim)
        
    def forward(self, x):
        # Input x is of shape (B, N, 131, 2)
        B, N, L, C = x.shape
        
        # Reshape x to be (B*N*131, 2) for individual landmark transformations
        x = x.view(B * N * L, C)
        
        # Apply the KAN layer
        x = self.kan_layer(x)
        
        # Reshape back to (B, N, 131, 2)
        x = x.view(B, N, L, C)
        
        return x