from ignite.metrics import Metric
import torch
from torch import nn
from scipy.stats import wasserstein_distance
from .utils import FACEMESH_ROI_IDX, FACEMESH_LIPS_IDX, FACEMESH_FACES_IDX
import numpy as np

mapped_lips_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_LIPS_IDX]
mapped_faces_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_FACES_IDX]

# Custom metric class
class CustomMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum = None
        self._num_examples = None
        super(CustomMetric, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._sum_fld = 0.0
        self._sum_flv = 0.0
        self._sum_mld = 0.0
        self._sum_mlv = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output[0].cpu() * 256., output[1].cpu() * 256.
        
        y_pred_faces = y_pred[:, :, mapped_faces_indices, :]
        y_faces = y[:, :, mapped_faces_indices, :]
        fld_score = self.calculate_LMD(y_pred_faces, y_faces)
        flv_score = self.calculate_LMV(y_pred_faces, y_faces)
        self._sum_fld += fld_score.sum()
        self._sum_flv += flv_score.sum()
            
        y_pred_lips = y_pred[:, :, mapped_lips_indices, :]
        y_lips = y[:, :, mapped_lips_indices, :]
        mld_score = self.calculate_LMD(y_pred_lips, y_lips)
        mlv_score = self.calculate_LMV(y_pred_lips, y_lips)
        self._sum_mld += mld_score.sum()
        self._sum_mlv += mlv_score.sum()
        
        self._num_examples = self._num_examples + y_pred.shape[0] + y_pred.shape[1]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomMetric must have at least one example before it can be computed')
        output = (f'[M-LD: {(self._sum_mld / self._num_examples):0.4f};'
                f'M-LV: {(self._sum_mlv / self._num_examples):0.4f};'
                f'F-LD: {(self._sum_fld / self._num_examples):0.4f};'
                f'F-LV: {(self._sum_flv / self._num_examples):0.4f}]'
        )
        return output
    
    def calculate_LMD(self, pred_landmark, gt_landmark, norm_distance=1.0):
        euclidean_distance = torch.sqrt(torch.sum((pred_landmark - gt_landmark)**2, dim=(pred_landmark.ndim - 1)))
        norm_per_frame = torch.mean(euclidean_distance, dim=(pred_landmark.ndim - 2))
        lmd = torch.divide(norm_per_frame, norm_distance)  
        return lmd
    
    def calculate_LMV(self, pred_landmark, gt_landmark, norm_distance=1.0):
        if gt_landmark.ndim == 4:
            velocity_pred_landmark = pred_landmark[:, 1:, :, :] - pred_landmark[:, 0:-1, :, :]
            velocity_gt_landmark = gt_landmark[:, 1:, :, :] - gt_landmark[:, 0:-1, :, :]
        elif gt_landmark.ndim == 3:
            velocity_pred_landmark = pred_landmark[1:, :, :] - pred_landmark[0:-1, :, :]
            velocity_gt_landmark = gt_landmark[1:, :, :] - gt_landmark[0:-1, :, :]
                
        euclidean_distance = torch.sqrt(torch.sum((velocity_pred_landmark - velocity_gt_landmark)**2, dim=(pred_landmark.ndim - 1)))
        norm_per_frame = torch.mean(euclidean_distance, dim=(pred_landmark.ndim - 2))
        lmv = torch.div(norm_per_frame, norm_distance)
        return lmv
    

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

