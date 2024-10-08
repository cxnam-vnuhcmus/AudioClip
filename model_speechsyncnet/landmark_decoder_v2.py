import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkDecoder(nn.Module):
    def __init__(self, input_dim=(5,128), hidden_dim=256, output_dim=(131,2)):
        super(LandmarkDecoder, self).__init__()

        input_size = input_dim[0] * input_dim[1]
        output_size = output_dim[0] * output_dim[1]  
        self.lstm = nn.LSTM(input_size=input_dim[1], hidden_size=hidden_dim, num_layers=2, bias=True, batch_first=False, dropout=0.1, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid()
        )

    def forward(self, audio_feature, landmark_feature):
        assert audio_feature.shape == landmark_feature.shape, "audio_feature and landmark_feature must have the same shape"
        
        combined_feature = audio_feature + landmark_feature  # (B, N, 128)        
        x = self.lstm(combined_feature)
        x = self.fc(x)  # (B, 132*2)
            
        out = x.reshape(x.shape[0],-1,2)
        return ouot
