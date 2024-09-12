import torch
import torch.nn as nn
from .utils import FACEMESH_ROI_IDX, FACEMESH_LIPS_IDX, FACEMESH_FACES_IDX

class LandmarkDecoder(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4, mouth_points=40, non_mouth_points=91):
        super(LandmarkDecoder, self).__init__()
        self.mouth_points = mouth_points
        self.non_mouth_points = non_mouth_points
        self.feature_dim = feature_dim
        self.num_heads = num_heads  # Số heads cho MHA
        
        # Multi-Head Attention layers
        self.mha1 = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=self.num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=self.num_heads, batch_first=True)
        self.mha3 = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=self.num_heads, batch_first=True)
        
        # Linear layers để chuyển đổi kích thước
        # self.linear_m1 = nn.Sequential(
        #     nn.Linear(self.feature_dim, self.feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_dim, self.feature_dim)
        # )
        
        self.linear_f1 = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # self.linear_m2 = nn.Sequential(
        #     nn.Linear(self.feature_dim, self.feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.feature_dim, self.mouth_points * 2),
        #     nn.Sigmoid()
        # )
        
        self.linear_f2 = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.non_mouth_points * 2),
            # nn.Sigmoid()
        )


    def forward(self, audio_feature, llfs_feature, lm_feature, gt_lm):
        # Step 1: Multi-Head Attention (MHA1)
        lm, _ = self.mha1(lm_feature, audio_feature, audio_feature)  # shape (B, 1, 128) -> (B, 1, 128)
        
        # Step 2: Chuyển đổi lm thành m1 và f1
        # m1 = self.linear_m1(lm)  # shape (B, 1, 128) -> (B, 1, 128)
        f1 = self.linear_f1(lm)  # shape (B, 1, 128) -> (B, 1, 128)
        
        # Step 3: Multi-Head Attention (MHA2 và MHA3)
        # m2, _ = self.mha2(m1, llfs_feature, llfs_feature)  # shape (B, 1, 128) -> (B, 1, 128)
        f2, _ = self.mha3(f1, llfs_feature, llfs_feature)  # shape (B, 1, 128) -> (B, 1, 128)
        
        # mouth_landmarks = self.linear_m2(m2)  # shape (B, 1, 128) -> (B, 1, 80)
        non_mouth_landmarks = self.linear_f2(f2)  # shape (B, 1, 128) -> (B, 1, 182)
        
        
        # mouth_landmarks = mouth_landmarks.view(mouth_landmarks.shape[0], mouth_landmarks.shape[1], 40, 2)
        non_mouth_landmarks = non_mouth_landmarks.view(non_mouth_landmarks.shape[0], non_mouth_landmarks.shape[1], 91, 2)
                
        # Combine mouth and non-mouth landmarks according to FACEMESH_ROI_IDX
        # full_landmarks = torch.zeros(mouth_landmarks.shape[0], 1, len(FACEMESH_ROI_IDX), 2, device=mouth_landmarks.device)  # (B, 1, 131, 2)
        full_landmarks = gt_lm.clone().unsqueeze(1)
        
        # Place mouth landmarks at positions specified by FACEMESH_LIPS_IDX
        # for i, idx in enumerate(FACEMESH_LIPS_IDX):
        #     full_landmarks[:, :, FACEMESH_ROI_IDX.index(idx), :] = mouth_landmarks[:, :, i, :]
        
        # Place non-mouth landmarks at positions specified by FACEMESH_FACES_IDX
        for i, idx in enumerate(FACEMESH_FACES_IDX):
            full_landmarks[:, :, FACEMESH_ROI_IDX.index(idx), :] = non_mouth_landmarks[:, :, i, :]
        
        return full_landmarks
