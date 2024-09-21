import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=128, output_dim=11):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Sequence(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
