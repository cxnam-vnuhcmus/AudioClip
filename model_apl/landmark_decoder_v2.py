import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkDecoder(nn.Module):
    def __init__(self, input_dim=(5,128), hidden_dim=256, output_dim=(131,2)):
        super(LandmarkDecoder, self).__init__()

        input_size = input_dim[0] * input_dim[1]
        output_size = output_dim[0] * output_dim[1]  
        self.linear1 = nn.Linear(input_size, hidden_dim)        
        self.linear2 = nn.Linear(hidden_dim, output_size)
        self.shortcut = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio_feature, landmark_feature):
        assert audio_feature.shape == landmark_feature.shape, "audio_feature and landmark_feature must have the same shape"
        
        combined_feature = audio_feature + landmark_feature  # (B, N, 128)        
        combined_feature = combined_feature.reshape(combined_feature.shape[0], -1)
        
        x = self.linear1(combined_feature)  # (B, hidden_dim)
        x = F.relu(x)  
        x = self.linear2(x)  # (B, output_dim)
        
        shortcut = self.shortcut(combined_feature)  # (B, output_dim)    
        output = x + shortcut  # (B, output_dim)

        output = self.norm(output)
        output = self.sigmoid(output)        
        output = output.reshape(output.shape[0],-1,2)
        
        return output
