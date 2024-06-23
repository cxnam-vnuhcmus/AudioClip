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
                 pretrained: Union[bool, str] = True,
                 img_dim: int,
                 lm_dim: int):
        super().__init__()
        
        self.pretrained = pretrained
        self.img_dim = img_dim
        self.lm_dim = lm_dim
        
        self.phoneme = PhonemeEncoder()
        self.landmark = LandmarkEncoder()
        # self.visual = FaceImageEncoder(pretrained=True)
        
        self.attention = CrossAttention(query_dim=self.img_dim * self.img_dim, context_dim=2*self.lm_dim)

        if isinstance(self.pretrained, str):
            self.load_state_dict(torch.load(self.pretrained, map_location='cpu'), strict=False)

        
        self.logit_scale_pl = torch.nn.Parameter(torch.log(torch.ones([]) * 100))

    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def encode_phoneme(self, phonemes: torch.Tensor) -> torch.Tensor:
        phoneme_embedding = self.phoneme(phonemes)
        return phoneme_embedding.to(self.device)
    
    def encode_landmark(self, landmarks: torch.Tensor) -> torch.Tensor:
        return self.landmark(landmarks.to(self.device))
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images.to(self.device))

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
        
        print(mask_features.shape)    
        print(ref_features.shape)    
        mask_features_view = mask_features.view(mask_features.shape[0], mask_features.shape[1], -1)
        ref_features_view = ref_features.view(ref_features.shape[0], ref_features.shape[1], -1)
        print(mask_features_view.shape)    
        print(ref_features_view.shape)    
        query_features = torch.stack((mask_features_view, ref_features_view), dim=1)
        print(query_features.shape)    
        context_features = torch.cat((phoneme_features, landmark_features), dim=1)
        print(context_features.shape)    
        
        output = self.attention(query_features, context_features)
        
        print(output.shape)
        
        logit_scale_pl = torch.clamp(self.logit_scale_pl.exp(), min=1.0, max=100.0)

        logits_phoneme_landmark = None

        if (phoneme_features is not None) and (landmark_features is not None):
            logits_phoneme_landmark = logit_scale_pl * phoneme_features @ landmark_features.T

        loss = self.loss_fn(logits_phoneme_landmark)

        return (phoneme_features, landmark_features, mask_features, ref_features, visual_features, visual), loss

    def loss_fn(self, logits_phoneme_landmark):
        batch_size = logits_phoneme_landmark.shape[0]

        reference = torch.arange(
            batch_size,
            dtype=torch.int64,
            device=self.device
        )

        loss = torch.tensor(0.0, dtype=torch.int64, device=self.device)

        num_modalities: int = 0
        scale = torch.tensor(1.0, dtype=torch.int64, device=self.device)
        
        loss_pl = F.cross_entropy(
            logits_phoneme_landmark, reference
        ) + F.cross_entropy(
            logits_phoneme_landmark.transpose(-1, -2), reference
        )
        loss = loss + loss_pl
        num_modalities += 1

        for idx in range(num_modalities):
            scale = scale * (idx + 1)

        return loss / scale

    @property
    def loss_fn_name(self) -> str:
        return 'Cross Entropy'
        
    def training_step_imp(model, batch, device) -> torch.Tensor:
        phoneme, landmark, mask, ref, visual_feat, visual = batch

        _, loss = model(
            phoneme = phoneme, 
            landmark = landmark,
            mask = mask,
            ref = ref,
            visual_feat = visual_feat,
            visual = visual,
        )

        return loss

    def eval_step_imp(model, batch, device, eval_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            phoneme, landmark, mask, ref, visual_feat, visual = batch

            (phoneme_features, landmark_features, mask_features, ref_features, visual_features, visual), _ = model(
                phoneme = phoneme, 
                landmark = landmark,
                mask = mask,
                ref = ref,
                visual_feat = visual_feat,
                visual = visual,
            )
            
        return phoneme_features, landmark_features
