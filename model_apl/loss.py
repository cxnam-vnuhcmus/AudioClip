from ignite.metrics import Metric
import torch

FACEMESH_ROI_IDX = [0, 4, 5, 6, 7, 13, 14, 17, 33, 37, 39, 40, 45, 46, 48, 52, 53, 55, 58, 61, 63, 65, 66, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 105, 107, 115, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 220, 234, 246, 249, 263, 267, 269, 270, 275, 276, 278, 282, 283, 285, 288, 291, 293, 295, 296, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 334, 336, 344, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 440, 454, 466]
FACEMESH_LIPS_IDX = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
mapped_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_LIPS_IDX]

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
        fld_score = self.calculate_LMD(y_pred, y)
        flv_score = self.calculate_LMV(y_pred, y)
        for fld in fld_score:
            self._sum_fld += fld
        for flv in flv_score:
            self._sum_flv += flv
            
        y_pred_lips = y_pred[:, mapped_indices, :]
        y_lips = y[:, mapped_indices, :]
        mld_score = self.calculate_LMD(y_pred_lips, y_lips)
        mlv_score = self.calculate_LMV(y_pred_lips, y_lips)
        for mld in mld_score:
            self._sum_mld += mld
        for mlv in mld_score:
            self._sum_mlv += mlv
        
        self._num_examples += y_pred.shape[0]

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