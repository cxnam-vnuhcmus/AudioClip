import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LandmarkToImageFeatureEncoder(nn.Module):
    def __init__(self):
        super(LandmarkToImageFeatureEncoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 478, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 4 * 32 * 32)
        
    def forward(self, x):
        # Input shape (B, 478, 2) -> (B, 2, 478) for Conv1D processing
        x = x.permute(0, 2, 1)
        
        # Convolutional layers with BatchNorm and ReLU
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        # Permute to (B, 478, 256) for Multi-head attention
        x = x.permute(0, 2, 1)

        # Apply Multi-head attention (B, 478, 256)
        attn_output, attn_weights = self.mha(x, x, x)

        # Flatten and fully connected layers
        x = self.flatten(attn_output)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # Reshape to (B, 4, 32, 32)
        x = x.view(-1, 4, 32, 32)
        return x
