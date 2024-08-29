import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** -0.5  # Scaling factor for dot product attention

    def forward(self, audio_feat, lm_feat):
        query = self.query_layer(audio_feat)
        key = self.key_layer(lm_feat)
        value = self.value_layer(lm_feat)
        
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) * self.scale, dim=-1)
        fused_feat = torch.matmul(attention_weights, value)
        
        return fused_feat

class LandmarkDecoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=131*2):
        super(LandmarkDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Attention-based fusion layer
        self.attention_fusion = AttentionFusion(input_dim)
        
        # Fully connected layers to process fused features
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization to stabilize training
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, audio_feat, lm_feat):
        # Apply attention-based fusion
        fused_feat = self.attention_fusion(audio_feat, lm_feat)
        
        # Pass the fused features through fully connected layers
        x = F.relu(self.fc_layers[0](fused_feat))
        x = self.layer_norm1(x)
        x = F.relu(self.fc_layers[2](x))
        x = self.layer_norm2(x)
        
        # Final output layer to predict landmarks
        out = self.fc_layers[4](x)
        out = out.view(-1, 131, 2)  # Reshape output to (B, 131, 2)
        
        return out
