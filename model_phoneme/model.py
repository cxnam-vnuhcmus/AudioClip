import os
import torch
import torch.nn.functional as F
from torch import nn

from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

from model_phoneme.phoneme_encoder import PhonemeEncoder
from model_phoneme.landmark_encoder import LandmarkEncoder
# from model_phoneme.visual_encoder import FaceImageEncoder
from model_phoneme.attention import CrossAttention

class Model(nn.Module):

    def __init__(self,
                 img_dim: int,
                 lm_dim: int,
                 pretrained: Union[bool, str] = True
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.img_dim = img_dim
        self.lm_dim = lm_dim
        
        self.phoneme = PhonemeEncoder()
        self.landmark = LandmarkEncoder()
        # self.visual = FaceImageEncoder(pretrained=True)
        
        self.attention = CrossAttention(query_dim=self.img_dim * self.img_dim, context_dim=self.lm_dim)
        # self.post_conv = nn.Sequential(
        #     nn.Conv2d(8, 4, 5, padding=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(4, 4, 3, padding=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(inplace=True)
        # )
        self.postconv = nn.Linear(1024,512)
        # self.postconv = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)

        if isinstance(self.pretrained, str):
            self.load_state_dict(torch.load(self.pretrained, map_location='cpu'), strict=False)

        
        self.logit_scale_pl = torch.nn.Parameter(torch.log(torch.ones([]) * 100))

    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def encode_phoneme(self, phonemes: torch.Tensor) -> torch.Tensor:
        tokens = self.phoneme.tokenizer(phonemes)
        tokens = tokens.to(self.device)
        phoneme_embedding = self.phoneme(tokens)
        return phoneme_embedding
    
    def encode_landmark(self, landmarks: torch.Tensor) -> torch.Tensor:
        landmarks = landmarks.to(self.device)
        return self.landmark(landmarks)
    
    # def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
    #     return self.visual(images.to(self.device))

    def forward(self,
                phoneme: Optional[torch.Tensor] = None,
                landmark: Optional[torch.Tensor] = None,
                mask_features: Optional[torch.Tensor] = None,
                ref_features: Optional[torch.Tensor] = None,
                visual_features: Optional[torch.Tensor] = None,
                visual: Optional[torch.Tensor] = None
                ):

        phoneme_features = None
        landmark_features = None
        if phoneme is not None:
            phoneme_features = self.encode_phoneme(phoneme)
            phoneme_features = phoneme_features / phoneme_features.norm(dim=-1, keepdim=True)
        if landmark is not None:
            landmark_features = self.encode_landmark(landmark)
            landmark_features = landmark_features / landmark_features.norm(dim=-1, keepdim=True)
        
        mask_features.to(self.device)
        ref_features.to(self.device)
        mask_features_view = mask_features.view(mask_features.shape[0], mask_features.shape[1], -1)
        ref_features_view = ref_features.view(ref_features.shape[0], ref_features.shape[1], -1)
        query_features = torch.cat((mask_features_view, ref_features_view), dim=1)
        context_features = torch.stack((phoneme_features, landmark_features), dim=1)
        
        output = self.attention(query_features, context_features)   #[2,8,32*32]
        output = self.postconv(output)                             #[2,8,512]
        pred_features = output.view(output.shape[0], 4, 32, 32)
        
        logit_scale_pl = torch.clamp(self.logit_scale_pl.exp(), min=1.0, max=100.0)
        logits_phoneme_landmark = None
        if (phoneme_features is not None) and (landmark_features is not None):
            logits_phoneme_landmark = logit_scale_pl * phoneme_features @ landmark_features.T

        loss = self.loss_fn(logits_phoneme_landmark, pred_features, visual_features)

        return (pred_features, visual_features, visual), loss

    def loss_fn(self, logits_phoneme_landmark, pred_features, gt_features):
        batch_size = logits_phoneme_landmark.shape[0]

        reference = torch.arange(
            batch_size,
            dtype=torch.int64,
            device=self.device
        )

        loss = torch.tensor(0.0, dtype=torch.int64, device=self.device)

        #CLIP loss
        loss_pl = F.cross_entropy(
            logits_phoneme_landmark, reference
        ) + F.cross_entropy(
            logits_phoneme_landmark.transpose(-1, -2), reference
        )
        
        #L1 loss
        mae_loss = F.l1_loss(pred_features, gt_features)
        
        #Cosine Similarity loss
        pred_flat = pred_features.view(batch_size, -1)
        gt_flat = gt_features.view(batch_size, -1)
        cos_sim_loss = F.cosine_similarity(pred_flat, gt_flat, dim=1)
        
        loss = loss_pl * 0.2 + mae_loss * 0.4 + cos_sim_loss * 0.4

        return loss

    @property
    def loss_fn_name(self) -> str:
        return 'Cross Entropy'
        
    def training_step_imp(model, batch, device) -> torch.Tensor:
        phoneme, landmark, mask, ref, visual_feat, visual = batch

        _, loss = model(
            phoneme = phoneme, 
            landmark = landmark,
            mask_features = mask,
            ref_features = ref,
            visual_features = visual_feat,
            visual = visual,
        )

        return loss

    def eval_step_imp(model, batch, device, eval_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            phoneme, landmark, mask, ref, visual_feat, visual = batch

            (pred_features, visual_features, visual), _ = model(
                phoneme = phoneme, 
                landmark = landmark,
                mask_features = mask,
                ref_features = ref,
                visual_features = visual_feat,
                visual = visual,
            )
            
        return pred_features, visual_features
