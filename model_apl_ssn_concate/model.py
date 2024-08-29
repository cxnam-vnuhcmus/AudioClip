import torch
from torch import nn
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
from typing import Union

from .audio_encoder import AudioEncoder
from .landmark_encoder import LandmarkEncoder
from .landmark_decoder import LandmarkDecoder
from .loss import CustomLoss
from .utils import plot_landmark_connections, calculate_LMD
from .utils import FACEMESH_ROI_IDX, FACEMESH_LIPS_IDX, FACEMESH_FACES_IDX
mapped_lips_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_LIPS_IDX]
mapped_faces_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_FACES_IDX]

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
        self.landmark = LandmarkEncoder(input_size=(131, 2), output_size=128, hidden_size=256)
        self.decoder = LandmarkDecoder(output_dim=self.lm_dim)
        
        self.criterion = CustomLoss(alpha=1.0, beta=0.5, gamma=0.5)

    
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio_embedding,_ = self.audio(audio.to(device=self.device, dtype=torch.float32))
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
        lm_loss = self.loss_fn(pred_lm, gt_lm)
        loss = lm_loss
        
        return (pred_lm), loss
        
    def loss_fn(self, pred_features, gt_features):
        loss = self.criterion(pred_features, gt_features)

        return loss

        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        audio, landmark, _ = batch
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
            audio, landmark, _ = batch
            audio = audio.to(device)
            landmark = landmark.to(device)
            gt_landmark_backup = landmark.clone()
            seg_len = (landmark.shape[1] + 1)//2
            for i in range(seg_len - 1):
                audio_seg = audio[:,i:i+seg_len]
                prv_landmark = landmark[:,i:i+seg_len-1]
                gt_landmark = gt_landmark_backup[:,i+seg_len-1]
                (pred_lm), _ = self(
                    audio = audio, 
                    landmark = prv_landmark,
                    gt_lm = gt_landmark
                )
                landmark[:,i+seg_len-1,:,:] = pred_lm
                
        return {"y_pred": landmark[:,seg_len-1:], "y": gt_landmark_backup[:,seg_len-1:]}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            audio, landmark, lm_paths = batch
            seg_len = (landmark.shape[1] + 1)//2
            prv_landmark = landmark[:,:seg_len-1]
            gt_landmark = landmark[:,seg_len-1]
            (pred_landmark), _ = self(
                audio = audio, 
                landmark = prv_landmark,
                gt_lm = gt_landmark
            )
            pred_landmark = pred_landmark.detach().cpu()
            
            for i in range(gt_landmark.shape[0]):
                image_size = 256
                output_file = os.path.join(save_folder, f'landmarks_{i}.png')
                gt_lm = gt_landmark[i]
                pred_lm = pred_landmark[i]
                
                y_pred_faces = pred_lm[mapped_faces_indices, :] * 256.
                y_faces = gt_lm[mapped_faces_indices, :] * 256.
                fld_score = calculate_LMD(y_pred_faces, y_faces)
                    
                y_pred_lips = pred_lm[mapped_lips_indices, :] * 256.
                y_lips = gt_lm[mapped_lips_indices, :] * 256.
                mld_score = calculate_LMD(y_pred_lips, y_lips)

                # Chuyển đổi tọa độ landmark từ khoảng (0, 1) về kích thước ảnh (0, image_size)
                gt_lm = gt_lm * image_size
                pred_lm = pred_lm * image_size

                # Tạo ảnh nền từ ảnh có sẵn cho cả ba phần
                img_paths = lm_paths[i].replace(".json", ".jpg")
                img_paths = img_paths.replace("face_meshes", "images")
                background = cv2.imread(img_paths)
                background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
                background = cv2.resize(background, (image_size, image_size))

                # Tạo một ảnh lớn hơn để chứa ba phần
                combined_image = np.ones((image_size, image_size * 3, 3), dtype=np.uint8) * 255

                # Copy ảnh nền vào ba phần của ảnh lớn
                combined_image[:, :image_size, :] = background
                combined_image[:, image_size:image_size*2, :] = background
                # combined_image[:, image_size*2:image_size*3, :] = background

                # Tạo subplots
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                # Phần 1: Ảnh background + Ground Truth
                axes[0].imshow(combined_image[:, :image_size, :])
                plot_landmark_connections(axes[0], gt_lm, 'green')
                axes[0].set_title('Ground Truth')
                axes[0].axis('off')

                # Phần 2: Ảnh background + Prediction
                axes[1].imshow(combined_image[:, image_size:image_size*2, :])
                plot_landmark_connections(axes[1], pred_lm, 'red')
                axes[1].set_title('Prediction')
                axes[1].axis('off')

                # Phần 3: Ảnh Ground Truth (đỏ) và Prediction (xanh dương)
                axes[2].imshow(combined_image[:, image_size*2:image_size*3, :])
                axes[2].scatter(gt_lm[:, 0], gt_lm[:, 1], color='green', label='Ground Truth', s=2)
                axes[2].scatter(pred_lm[:, 0], pred_lm[:, 1], color='red', label='Prediction', s=2)
                # axes[2].set_title('GT (Green) vs Prediction (Red)')
                axes[2].set_title(f'[M-LD: {mld_score:0.4f};F-LD: {fld_score:0.4f};]')
                axes[2].axis('off')

                # Lưu ảnh vào file
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()
 