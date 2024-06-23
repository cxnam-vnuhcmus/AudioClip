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
from model_phoneme.visual_encoder import FaceImageEncoder
from model_phoneme.attention import CrossAttention

class Model(nn.Module):

    def __init__(self,
                 embed_dim: int = 1024,
                 pretrained: Union[bool, str] = True):
        super().__init__()
        
        self.pretrained = pretrained
        
        self.phoneme = PhonemeEncoder()
        self.landmark = LandmarkEncoder()
        self.visual = FaceImageEncoder(pretrained=True)
        
        self.attention = CrossAttention(query_dim=128, context_dim=128)

        if isinstance(self.pretrained, str):
            self.load_state_dict(torch.load(self.pretrained, map_location='cpu'), strict=False)

        self.embed_dim = embed_dim
        
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
                visual: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                ref: Optional[torch.Tensor] = None
                ):

        phoneme_features = None
        landmark_features = None
        visual_features = None
        mask_features = None
        ref_features = None
        
        if phoneme is not None:
            phoneme_features = self.encode_phoneme(phoneme)
            phoneme_features = phoneme_features / phoneme_features.norm(dim=-1, keepdim=True)
        if landmark is not None:
            landmark_features = self.encode_landmark(landmark)
            landmark_features = landmark_features / landmark_features.norm(dim=-1, keepdim=True)
        if visual is not None:
            visual_features = self.encode_visual(visual)
            visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        if mask is not None:
            mask_features = self.encode_visual(mask)
            mask_features = mask_features / mask_features.norm(dim=-1, keepdim=True)
        if ref is not None:
            ref_features = self.encode_visual(ref)
            ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
            
        query_features = torch.cat((phoneme_features, landmark_features), dim=1)
        context_features = torch.cat((mask_features, ref_features), dim=1)
        
        output = self.attention(query_features, context_features)
        
        logit_scale_pl = torch.clamp(self.logit_scale_pl.exp(), min=1.0, max=100.0)

        logits_phoneme_landmark = None

        if (phoneme_features is not None) and (landmark_features is not None):
            logits_phoneme_landmark = logit_scale_pl * phoneme_features @ landmark_features.T

        loss = self.loss_fn(logits_phoneme_landmark)

        return (phoneme_features, landmark_features, visual_features, mask_features, ref_features), loss

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
        phoneme, landmark, visual, mask, ref = batch

        _, loss = model(
            phoneme = phoneme, 
            landmark = landmark,
            visual = visual,
            mask = mask,
            ref = ref
        )

        return loss

    def eval_step_imp(model, batch, device, eval_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            phoneme, landmark, visual, mask, ref = batch

            (phoneme_features, landmark_features, visual_features, mask_features, ref_features), _ = model(
                phoneme = phoneme, 
                landmark = landmark,
                visual = visual,
                mask = mask,
                ref = ref
            )
            
        return phoneme_features, landmark_features
