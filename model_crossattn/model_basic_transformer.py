import os
import torch
import torch.nn.functional as F
from torch import nn

from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

from model_attn.audio_encoder import AudioEncoder
from model_attn.visual_encoder import VisualEncoder
from model_attn.attention_diffusers import BasicTransformerBlock

import numpy as np
import matplotlib.pyplot as plt

class Model(nn.Module):

    def __init__(self,
                 img_dim: int,
                 lm_dim: int,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        self.img_dim = img_dim
        self.lm_dim = lm_dim
        
        self.audio = AudioEncoder(output_dim=32*32)
        self.visual = AudioEncoder(output_dim=32*32)
        self.cross_attn = BasicTransformerBlock(
            dim= 32*32,
            num_attention_heads= 8,
            attention_head_dim= 512,
            cross_attention_dim= 32*32)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32*32, 32*32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32*32),
            nn.Linear(32*32, 32*32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32*32),
            nn.Linear(32*32, 16*16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16*16),
            nn.Linear(16*16, 16*16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16*16),
            nn.Linear(16*16, 8),
        )

    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio_embedding = self.audio(audio.to(self.device))
        return audio_embedding
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images.to(self.device))
    
    def forward(self,
                audio,
                visual,
                emotion,
                grad_map = False,
                save_folder = ""
                ):
        audio_features = self.encode_audio(audio)
        visual_features = self.encode_visual(visual)
        attn_features = self.cross_attn(
            hidden_states=visual_features.unsqueeze(1), 
            encoder_hidden_states=audio_features.unsqueeze(1))
        
        attn_features = self.flatten(attn_features)

        out_features = self.fc(attn_features)
        
        loss = self.loss_fn(out_features, emotion)
        
        with torch.no_grad():
            if grad_map:
                attn_map = attn_features.clone()
                attn_map = attn_map.detach().cpu().numpy()
                attn_map = attn_map.reshape(attn_map.shape[0], 1, 32, 32)
                for i in range(attn_map.shape[0]):
                    heatmap = attn_map[i].squeeze(0)
                    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
                    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
                    plt.axis('off')  # Tắt các trục để chỉ hiển thị heatmap
                    plt.savefig(f'{save_folder}/image_{i:05d}.jpg', bbox_inches='tight', pad_inches=0)  # Lưu heatmap với định dạng JPG
                    plt.close()  # Đóng plot để giải phóng bộ nhớ

        return (out_features), loss
        
    def loss_fn(self, pred_features, gt_features):
        gt_tensor_indices = torch.argmax(gt_features, dim=1)
        
        loss = nn.CrossEntropyLoss()(pred_features, gt_tensor_indices)

        return loss

        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        phoneme, audio, visual, emotion = batch
        emotion = emotion.to(self.module.device)

        _, loss = self(
            audio = audio, 
            visual = visual,
            emotion = emotion
        )
        
        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            phoneme, audio, visual, emotion = batch
            emotion = emotion.to(self.module.device)
            
            (pred_feature), _ = self(
                audio = audio, 
                visual = visual,
                emotion = emotion
            )
            
            gt_tensor_indices = torch.argmax(emotion, dim=1)
            
            print(pred_feature.shape)
            print(emotion.shape)
            print(gt_tensor_indices.shape)
        return {"y_pred": pred_feature, "y": emotion, "y_indices": gt_tensor_indices}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            phoneme, audio, visual, emotion = batch
            emotion = emotion.to(self.module.device)
            
            _, _ = self(
                audio = audio, 
                visual = visual,
                emotion = emotion,
                grad_map = True,
                save_folder = save_folder
            )