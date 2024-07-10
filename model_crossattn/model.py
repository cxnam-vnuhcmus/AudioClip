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
from model_attn.attention import SpatialTransformer

import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

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
        
        self.audio = AudioEncoder(output_dim=512)
        self.visual = VisualEncoder(attention_head_dim=64, context_dim=512, num_classes=8)
    
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio_embedding = self.audio(audio.to(device=self.device, dtype=torch.float32))
        return audio_embedding
    
    def encode_visual(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.visual(images.to(device=self.device, dtype=torch.float32), context)
    
    def forward(self,
                audio,
                visual,
                emotion,
                grad_map = False,
                save_folder = ""
                ):
        audio_features = self.encode_audio(audio)
        out_features, (x1,x2,x3,x4) = self.encode_visual(visual, context=audio_features)
        

        out_features = out_features.to(self.device)
        emotion = emotion.to(self.device)
        loss = self.loss_fn(out_features, emotion)
        
        with torch.no_grad():
            if grad_map:
                attn_map = x1.clone()
                attn_map = torch.sigmoid(attn_map)
                attn_map = attn_map.detach().cpu().numpy()
                for i in range(attn_map.shape[0]):
                    # heatmap = (attn_map[i][0] * 255).astype(np.uint8)
                    heatmap = attn_map[i][0]
                    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255

                    plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
                    plt.axis('off') 
                    plt.savefig(f'{save_folder}/image_{i:05d}.jpg', bbox_inches='tight', pad_inches=0) 
                    plt.close()  

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