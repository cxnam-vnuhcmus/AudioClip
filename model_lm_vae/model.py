import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
from typing import Union
import json

from .landmark_encoder_v2 import LandmarkToImageFeatureEncoder
from .loss import CustomLoss
from .utils import plot_landmark_connections, calculate_LMD
from .utils import FACEMESH_ROI_IDX, FACEMESH_LIPS_IDX, FACEMESH_FACES_IDX
mapped_lips_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_LIPS_IDX]
mapped_faces_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_FACES_IDX]


class Model(nn.Module):

    def __init__(self,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        
        self.landmark = LandmarkToImageFeatureEncoder()
        
        self.criterion = nn.MSELoss()

    
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def encode_landmark(self, landmarks: torch.Tensor, img_feature_mask) -> torch.Tensor:
        landmarks = landmarks.to(self.device)
        return self.landmark(landmarks, img_feature_mask)
        
    def forward(self,
                landmark,
                gt_img_feature,
                ):
        img_feature_mask = gt_img_feature.clone()
        img_feature_mask[:, :, 2*8:3*8, 1*8:3*8] = 1

        img_features = self.encode_landmark(landmark, img_feature_mask)        #(B,131,2) -> (B,4,32,32)
        
        loss = self.loss_fn(img_features, gt_img_feature)
        
        return (img_features), loss
        
    def loss_fn(self, pred_features, gt_features):
        loss = self.criterion(pred_features, gt_features)

        return loss

        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        landmark, gt_img_feature, _ = batch
        _, loss = self(
            landmark = landmark,
            gt_img_feature = gt_img_feature
        )
        
        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            landmark, gt_img_feature, _ = batch
            gt_img_feature = gt_img_feature.to(device)
            
            (img_feature), _ = self(
                landmark = landmark,
                gt_img_feature = gt_img_feature
            )
            
                
        return {"y_pred": img_feature, "y": gt_img_feature}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            landmark, gt_img_feature, lm_paths = batch
            
            (img_feature), _ = self(
                landmark = landmark,
                gt_img_feature = gt_img_feature
            )
            
            gt_img_feature_list = gt_img_feature.tolist()
            img_feature_list = img_feature.tolist()
            data = {
                "gt_img_feature": gt_img_feature_list,
                "pred_img_feature": img_feature_list,
                "lm_paths": lm_paths
            }
            with open('./assets/samples/M003/samples_lm_vae/tensor_data.json', 'w') as json_file:
                json.dump(data, json_file)
            
            gt_img_feature = gt_img_feature.permute(0, 2, 3, 1)
            img_feature = img_feature.permute(0, 2, 3, 1).detach().cpu()
            
            for i in range(landmark.shape[0]):
                image_size = 32
                output_file = os.path.join(save_folder, f'landmarks_{i}.png')
                
                gt_lm = gt_img_feature[i][:,:,:3]
                pred_lm = img_feature[i][:,:,:3]

                combined_image = np.ones((image_size, image_size * 2, 3), dtype=np.uint8) * 255
                

                # Copy ảnh nền vào ba phần của ảnh lớn
                combined_image[:, :image_size, :] = gt_lm
                combined_image[:, image_size:, :] = pred_lm

                # Tạo subplots
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Phần 1: Ảnh background + Ground Truth
                axes[0].imshow(combined_image[:, :image_size, :])
                axes[0].set_title('Ground Truth')
                axes[0].axis('off')

                # Phần 2: Ảnh background + Prediction
                axes[1].imshow(combined_image[:, image_size:image_size*2, :])
                axes[1].set_title('Prediction')
                axes[1].axis('off')

                # Lưu ảnh vào file
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()