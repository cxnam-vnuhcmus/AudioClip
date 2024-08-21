import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder models and ContrastiveLoss as before
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, audio_features, landmark_features):
        audio_features = audio_features.squeeze(1)  # Shape: [32, 128]
        landmark_features = landmark_features.squeeze(1)  # Shape: [32, 128]
        audio_features = nn.functional.normalize(audio_features, p=2, dim=-1)
        landmark_features = nn.functional.normalize(landmark_features, p=2, dim=-1)
        similarity_matrix = torch.matmul(audio_features, landmark_features.T) / self.temperature
        labels = torch.arange(audio_features.size(0), device=audio_features.device)
        loss = nn.CrossEntropyLoss(reduction='none')(similarity_matrix, labels)
        return loss.detach().cpu()
    
class ContrastiveModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=128):
        super(ContrastiveModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create the Encoder models
        self.audio_encoder = Encoder(input_dim, hidden_dim, output_dim)
        self.landmark_encoder = Encoder(input_dim, hidden_dim, output_dim)

        # Create the Contrastive Loss function
        self.contrastive_loss_fn = ContrastiveLoss(temperature=0.07)


    def forward(self, audio_features, landmark_features):
        audio_embeddings = self.audio_encoder(audio_features)
        landmark_embeddings = self.landmark_encoder(landmark_features)
        
        # Compute the loss
        loss = self.contrastive_loss_fn(audio_embeddings, landmark_embeddings)

        return loss