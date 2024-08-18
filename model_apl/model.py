import torch
from torch import nn

from typing import Union

from model_apl.audio_encoder import AudioEncoder
from model_apl.landmark_encoder import LandmarkEncoder
from model_apl.landmark_decoder import LandmarkDecoder

import cv2
import numpy as np
import torchvision
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np
import os


# Chamfer Distance Function
def chamfer_distance(x, y):
    dist_matrix = torch.cdist(x, y)
    min_dist_x = torch.min(dist_matrix, dim=1)[0]
    min_dist_y = torch.min(dist_matrix, dim=0)[0]
    return torch.mean(min_dist_x) + torch.mean(min_dist_y)

# Earth Mover's Distance Function
def earth_mover_distance(pred_feat, gt_feat):
    pred_feat = pred_feat.detach().numpy()
    gt_feat = gt_feat.detach().numpy()
    B = pred_feat.shape[0]
    emd_scores = []

    for i in range(B):
        pred = pred_feat[i].reshape(-1)
        gt = gt_feat[i].reshape(-1)
        emd = wasserstein_distance(pred, gt)
        emd_scores.append(emd)

    return torch.tensor(np.array(emd_scores))

# Custom Loss Function Combining MAE, Chamfer Distance, and EMD
class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.cpu()
        target = target.cpu()
        
        # MAE Loss
        mae_loss = nn.L1Loss()(pred, target) #MAE Loss: Đo khoảng cách trung bình tuyệt đối giữa các điểm, giúp cải thiện độ chính xác của các tọa độ điểm.
        
        # Chamfer Distance Loss
        chamfer_loss = chamfer_distance(pred, target) #Chamfer Distance: Đo sự tương đồng giữa hai tập hợp điểm bằng cách tính khoảng cách giữa các điểm gần nhất, có thể giúp cải thiện cấu trúc của các điểm landmark.
        
        # Earth Mover's Distance Loss
        emd_loss = earth_mover_distance(pred, target) #EMD: Đo sự khác biệt giữa các phân phối điểm, giúp cải thiện khả năng phân phối của các điểm landmark.
        
        # Tổng hợp các mất mát với trọng số
        total_loss = self.alpha * mae_loss + self.beta * chamfer_loss + self.gamma * emd_loss
        return total_loss


class Model(nn.Module):

    def __init__(self,
                 audio_dim: int,
                 lm_dim: int,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        self.audio_dim = audio_dim
        self.lm_dim = lm_dim
        
        self.audio = AudioEncoder(dim_in=self.audio_dim)
        self.landmark = LandmarkEncoder(input_dim=self.lm_dim, hidden_dim=128, output_dim=128, num_heads=8, num_layers=3)
        self.decoder = LandmarkDecoder(output_dim=self.lm_dim)
        
        self.criterion = CustomLoss(alpha=1.0, beta=0.5, gamma=0.5)

    
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio_embedding, _ = self.audio(audio.to(device=self.device, dtype=torch.float32))
        return audio_embedding
    
    def encode_landmark(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = landmarks.to(self.device)
        return self.landmark(landmarks)
        
    def forward(self,
                audio,
                landmark,
                gt_lm
                ):
        audio_features = self.encode_audio(audio)                 #(B,N,80) -> (B,1,128)
        landmark_features = self.encode_landmark(landmark)        #(B,N-1,131,2) -> (B,1,128)
        pred_lm = self.decoder(audio_features, landmark_features) #(B,1,131,2)
        pred_lm = pred_lm.squeeze(1)
        loss = self.loss_fn(pred_lm, gt_lm)
        
        return (pred_lm), loss
        
    def loss_fn(self, pred_features, gt_features):
        loss = self.criterion(pred_features, gt_features)

        return loss

        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        audio, landmark = batch
        prv_landmark = landmark[:,:-1]
        gt_landmark = landmark[:,-1]
        _, loss = self(
            audio = audio, 
            landmark = prv_landmark,
            gt_lm = gt_landmark
        )
        
        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            audio, landmark = batch
            prv_landmark = landmark[:,:-1]
            gt_landmark = landmark[:,-1].to(device)
            (pred_lm), _ = self(
                audio = audio, 
                landmark = prv_landmark,
                gt_lm = gt_landmark
            )
        return {"y_pred": pred_lm, "y": gt_landmark}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            audio, landmark = batch
            prv_landmark = landmark[:,:-1]
            gt_landmark = landmark[:,-1]
            (pred_landmark), _ = self(
                audio = audio, 
                landmark = prv_landmark,
                gt_lm = gt_landmark
            )
            pred_landmark = pred_landmark.detach().cpu()
            
            for i in range(gt_landmark.shape[0]):
                image_size=256
                output_file=os.path.join(save_folder,f'landmarks_{i}.png')
                gt_lm = gt_landmark[i]
                pred_lm = pred_landmark[i]
                
                # Chuyển đổi tọa độ landmark từ khoảng (0, 1) về kích thước ảnh (0, image_size)
                gt_lm = gt_lm * image_size
                pred_lm = pred_lm * image_size
                
                # Tạo ảnh nền trắng
                image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
                
                # Vẽ các điểm landmark
                plt.figure(figsize=(8, 8))
                plt.imshow(image)
                
                # Vẽ landmark của ground truth với màu đỏ
                plt.scatter(gt_lm[:, 0], gt_lm[:, 1], color='red', label='Ground Truth', s=10)
                
                # Vẽ landmark của prediction với màu xanh dương
                plt.scatter(pred_lm[:, 0], pred_lm[:, 1], color='blue', label='Prediction', s=10)
                
                # Thiết lập tiêu đề và legend
                plt.title('Landmarks Visualization')
                plt.legend()
                
                # Lưu ảnh vào file
                plt.axis('off')
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()