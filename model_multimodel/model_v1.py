import torch
from torch import nn
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
from typing import Union

from .audio_encoder_v1 import AudioEncoder
from .landmark_encoder_v1 import LandmarkEncoder
from .landmark_decoder import LandmarkDecoder
from .loss import CustomLoss
from .contrastive import ContrastiveModel
from .utils import plot_landmark_connections, calculate_LMD
from .utils import FACEMESH_ROI_IDX, FACEMESH_LIPS_IDX, FACEMESH_FACES_IDX
mapped_lips_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_LIPS_IDX]
mapped_faces_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_FACES_IDX]
from .landmark_decoder import BottleneckBlock1D, KAN

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
        
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
        
        # self.fc1 = nn.Linear(80, 56)
        # self.fc2 = nn.Linear(32, 56)
        
        self.audio_mouth = AudioEncoder(dim_in=40)
        self.audio_face = AudioEncoder(dim_in=40)
        self.llfs = AudioEncoder(dim_in=32)
        self.landmark_mouth = LandmarkEncoder(input_dim=40* 2)
        self.landmark_face = LandmarkEncoder(input_dim=131* 2)
        self.decoder_mouth = LandmarkDecoder(input_dim=128+128, hidden_dim=128, output_dim=40*2)
        self.decoder_face = LandmarkDecoder(input_dim=128+128, hidden_dim=128, output_dim=131*2)
        self.decoder_llfs = LandmarkDecoder(input_dim=128+128, hidden_dim=128, output_dim=131*2)
        
        
        # self.bottleneck = BottleneckBlock1D(in_channels=2, bottleneck_channels=64, out_channels=2, residual=True)
        self.kan = KAN()
        
        self.criterion = CustomLoss(alpha=1.0, beta=0.5, gamma=0.5)
        
        
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio_embedding = self.audio(audio.to(device=self.device, dtype=torch.float32))
        return audio_embedding

    def encode_llfs(self, llfs: torch.Tensor) -> torch.Tensor:
        llfs_embedding = self.llfs(llfs.to(device=self.device, dtype=torch.float32))
        return llfs_embedding
    
    def encode_landmark(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = landmarks.to(self.device)
        return self.landmark(landmarks)
        
    def forward(self,
                audio,
                llfs,
                landmark,
                gt_lm
                ):
        audio = audio.to(self.device)
        llfs = llfs.to(self.device)
        landmark = landmark.to(self.device)
        lm_mouth = landmark[:,:,mapped_lips_indices,:]
        # lm_face = landmark[:,:,mapped_faces_indices,:]
        
        audio_mouth_features = self.audio_mouth(audio)
        audio_face_features = self.audio_face(audio)        
        llfs_features = self.llfs(llfs)
        
        landmark_mouth_features = self.landmark_mouth(lm_mouth)
        landmark_face_features = self.landmark_face(landmark)
                
        pred_mouth = self.decoder_mouth(audio_mouth_features, landmark_mouth_features) #(B,1,40,2)
        pred_face_1 = self.decoder_face(audio_face_features, landmark_face_features) #(B,1,40,2)
        pred_face_2 = self.decoder_llfs(llfs_features, landmark_face_features)
        
        pred_face = pred_face_1 + pred_face_2        
        # pred_face = self.bottleneck(pred_face)
        
        
        # full_landmarks = torch.zeros(pred_mouth.shape[0], 1, len(FACEMESH_ROI_IDX), 2, device=pred_mouth.device)  # (B, 1, 131, 2)
        
        # Place mouth landmarks at positions specified by FACEMESH_LIPS_IDX
        for i, idx in enumerate(FACEMESH_LIPS_IDX):
            pred_face[:, :, FACEMESH_ROI_IDX.index(idx), :] = pred_mouth[:, :, i, :]
        
        # Place non-mouth landmarks at positions specified by FACEMESH_FACES_IDX
        # for i, idx in enumerate(FACEMESH_FACES_IDX):
        #     full_landmarks[:, :, FACEMESH_ROI_IDX.index(idx), :] = pred_face[:, :, i, :]
        
        pred_face = self.kan(pred_face)
        pred_lm = pred_face.squeeze(1)
        lm_loss = self.loss_fn(pred_lm, gt_lm).to(self.device)
        loss = lm_loss 
        
        return (pred_lm), loss
        
    def loss_fn(self, pred_features, gt_features):
        loss = self.criterion(pred_features, gt_features)

        return loss

        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        audio, llfs, landmark, _ = batch
        prv_landmark = landmark[:,:-1]
        gt_landmark = landmark[:,-1]
        _, loss = self(
            audio = audio, 
            llfs = llfs,
            landmark = prv_landmark,
            gt_lm = gt_landmark
        )
        
        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            audio, llfs, landmark, _ = batch
            audio = audio.to(device)
            llfs = llfs.to(device)
            landmark = landmark.to(device)
            gt_landmark_backup = landmark.clone()
            seg_len = (landmark.shape[1] + 1)//2
            for i in range(seg_len - 1):
                audio_seg = audio[:,i:i+seg_len]
                llfs_seg = llfs[:,i:i+seg_len]
                prv_landmark = landmark[:,i:i+seg_len-1]
                gt_landmark = gt_landmark_backup[:,i+seg_len-1]
                (pred_lm), _ = self(
                    audio = audio_seg, 
                    llfs = llfs_seg,
                    landmark = prv_landmark,
                    gt_lm = gt_landmark
                )
                landmark[:,i+seg_len-1,:,:] = pred_lm
                
        return {"y_pred": landmark[:,seg_len-1:], "y": gt_landmark_backup[:,seg_len-1:]}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            audio, llfs, landmark, lm_paths = batch
            seg_len = (landmark.shape[1] + 1)//2
            audio_seg = audio[:,:seg_len]
            llfs_seg = llfs[:,:seg_len]
            prv_landmark = landmark[:,:seg_len-1]
            gt_landmark = landmark[:,seg_len-1]
            (pred_landmark), _ = self(
                audio = audio_seg,
                llfs = llfs_seg,
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
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={'wspace': 0, 'hspace': 0})

                # Phần 1: Ảnh background + Ground Truth
                axes[0].imshow(combined_image[:, :image_size, :])
                plot_landmark_connections(axes[0], gt_lm, 'green')
                # axes[0].set_title('Ground Truth')
                axes[0].axis('off')

                # Phần 2: Ảnh background + Prediction
                axes[1].imshow(combined_image[:, image_size:image_size*2, :])
                plot_landmark_connections(axes[1], pred_lm, 'red')
                # axes[1].set_title('Prediction')
                axes[1].axis('off')

                # Phần 3: Ảnh Ground Truth (đỏ) và Prediction (xanh dương)
                axes[2].imshow(combined_image[:, image_size*2:image_size*3, :])
                axes[2].scatter(gt_lm[:, 0], gt_lm[:, 1], color='green', label='Ground Truth', s=2)
                axes[2].scatter(pred_lm[:, 0], pred_lm[:, 1], color='red', label='Prediction', s=2)
                # axes[2].set_title('GT (Green) vs Prediction (Red)')
                # axes[2].set_title(f'[M-LD: {mld_score:0.4f};F-LD: {fld_score:0.4f};]')
                axes[2].axis('off')
                
                # Add text on top of the image
                title_text = f'[M-LD: {mld_score:0.4f}; F-LD: {fld_score:0.4f}]'
                axes[2].text(20, 20, title_text, fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=1.0))


                # Lưu ảnh vào file
                plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
                plt.close()
