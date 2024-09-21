import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, num_classes=8):
        super(EmotionClassifier, self).__init__()
        # Lớp fully connected đầu tiên (giảm kích thước xuống hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        # Lớp fully connected thứ hai (đầu ra cho số lớp cảm xúc)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Đưa đầu vào qua hai lớp fully connected
        x = x.view(x.size(0), -1)  # Chuyển input từ (B, 1, 128) thành (B, 128)
        x = self.fc1(x)            # FC1
        x = self.relu(x)           # Activation
        x = self.fc2(x)            # FC2 (output size = num_classes)
        return x