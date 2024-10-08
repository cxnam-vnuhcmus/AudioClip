
import torch
import torch.nn as nn

class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim=(131,2), hidden_dim=256, output_dim=128):
        super(LandmarkEncoder, self).__init__()
        
        input_size = input_dim[0] * input_dim[1]
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.fc(x)
        
        return x