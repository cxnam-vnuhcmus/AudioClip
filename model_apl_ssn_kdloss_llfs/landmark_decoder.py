import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkDecoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=131*2):
        super(LandmarkDecoder, self).__init__()
        self.output_dim = output_dim
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.shortcut = nn.Linear(input_dim, hidden_dim)
        
        self.projection1 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, audio_feature, landmark_feature, llfs_feature):
        assert audio_feature.shape == landmark_feature.shape, "audio_feature and landmark_feature must have the same shape"
        
        combined_feature = torch.cat((audio_feature,landmark_feature,llfs_feature), dim=-1)  
        
        x = self.linear1(combined_feature)  # (B, N, hidden_dim)
        x = F.relu(x)  # Áp dụng ReLU activation function
        x = self.linear2(x)  # (B, N, output_dim)
        
        shortcut = self.shortcut(combined_feature)  # (B, N, output_dim)
        
        output = x + shortcut  # (B, N, output_dim)
        
        output = self.projection1(output)
        
        return output
