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
from model_phoneme.visual_encoder import VisualEncoder
from model_phoneme.attention import CrossAttention
from torchvision.utils import save_image
from taming.modules.losses.lpips import LPIPS

import numpy as np

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
        
        self.phoneme = PhonemeEncoder()
        self.landmark = LandmarkEncoder()
        self.visual = VisualEncoder()
        
        self.attention = CrossAttention(query_dim=self.img_dim * self.img_dim, context_dim=self.lm_dim)

        self.linear = nn.Linear(8*self.img_dim*self.img_dim, 4*self.img_dim*self.img_dim)
        
        self.logit_scale_pl = torch.nn.Parameter(torch.log(torch.ones([]) * 100))
        
        self.logvar = nn.Parameter(torch.ones(size=()) * 1.0)
        
        self.perceptual_loss = LPIPS()

        for p in self.parameters():
            p.requires_grad = True
        for p in self.visual.vae.encoder.parameters():
            p.requires_grad = False
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

        # for name, param in self.named_parameters():
        #     print(f'Parameter name: {name}, Require gradient: {param.requires_grad}')


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
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual.encode(images.to(self.device))

    def decode_visual(self, latents: torch.Tensor) -> torch.Tensor:
        return self.visual.decode(latents)
    
    def forward(self,
                phoneme,
                landmark,
                mask,
                ref
                ):

        phoneme_features = self.encode_phoneme(phoneme)
        phoneme_features = phoneme_features / phoneme_features.norm(dim=-1, keepdim=True)

        landmark_features = self.encode_landmark(landmark)
        landmark_features = landmark_features / landmark_features.norm(dim=-1, keepdim=True)

        mask_features = self.encode_visual(mask)
        ref_features = self.encode_visual(ref)

        phoneme_features = nn.Parameter(phoneme_features, requires_grad=True)   
        landmark_features = nn.Parameter(landmark_features, requires_grad=True)   
        mask_features = nn.Parameter(mask_features, requires_grad=True)   
        ref_features = nn.Parameter(ref_features, requires_grad=True)   
        
        mask_features_view = mask_features.reshape((mask_features.shape[0], mask_features.shape[1], -1))
        ref_features_view = ref_features.reshape((ref_features.shape[0], ref_features.shape[1], -1))
        query_features = torch.cat((mask_features_view, ref_features_view), dim=1)
        context_features = torch.stack((phoneme_features, landmark_features), dim=1)
        
        output_att = self.attention(query_features, context_features)   #[2,8,32*32]
        output_att = output_att.view(output_att.shape[0], -1)            #[2,8*32*32]
        pred_features = self.linear(output_att)                              #[2,4*32*32] 
        recons_features = pred_features.reshape((pred_features.shape[0], -1, self.img_dim, self.img_dim))    
        
        reconstructed_images = self.decode_visual(recons_features)

        return (phoneme_features, landmark_features, pred_features, reconstructed_images)

    def nll(self, image, mean, logvar, dims=[1, 2, 3]):
        logtwopi = np.log(2.0 * np.pi)
        var = torch.exp(logvar)
        return 0.5 * torch.sum(
            logtwopi + logvar + torch.pow(image - mean, 2) / var,
            dim=dims,
        )
        
    def loss_fn(self, phoneme_features, landmark_features, pred_features, gt_features, reconstructed_images, gt_images):
        batch_size = phoneme_features.shape[0]

        #CLIP loss
        reference = torch.arange(
            batch_size,
            dtype=torch.int64,
            device=self.device
        )
        
        logit_scale_pl = torch.clamp(self.logit_scale_pl.exp(), min=1.0, max=100.0)
        logits_phoneme_landmark = None
        if (phoneme_features is not None) and (landmark_features is not None):
            logits_phoneme_landmark = logit_scale_pl * phoneme_features @ landmark_features.T

        loss_pl = nn.CrossEntropyLoss()(
            logits_phoneme_landmark, reference
        ) + nn.CrossEntropyLoss()(
            logits_phoneme_landmark.transpose(-1, -2), reference
        )
        
        # L2 loss
        mae_loss = nn.MSELoss()(pred_features, gt_features)
    
        # Cosine Similarity loss    
        cos_sim_loss = 1 - F.cosine_similarity(pred_features, gt_features, dim=1).mean()
        
        # Reconstruction loss
        rec_loss = torch.abs(gt_images.contiguous() - reconstructed_images.contiguous())
        p_loss = self.perceptual_loss(gt_images.contiguous(), reconstructed_images.contiguous())
        rec_loss = rec_loss + 1.0 * p_loss
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = 1.0 * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]

        #Total loss
        loss = loss_pl * 0.1
        + mae_loss * 0.4 
        + cos_sim_loss * 0.2 
        + weighted_nll_loss * 0.4

        return loss

        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        phoneme, landmark, mask, ref, gt = batch

        (phoneme_features, landmark_features, pred_features, reconstructed_images) = self(
            phoneme = phoneme, 
            landmark = landmark,
            mask = mask,
            ref = ref
        )
        
        visual_features = self.module.encode_visual(gt)
        visual_features = visual_features.reshape((visual_features.shape[0], -1))

        gt = gt.to(self.module.device)
        loss = self.module.loss_fn(phoneme_features, landmark_features, pred_features, visual_features, reconstructed_images, gt)

        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            phoneme, landmark, mask, ref, gt = batch
            
            (_, _, pred_features, _) = self(
                phoneme = phoneme, 
                landmark = landmark,
                mask = mask,
                ref = ref
            )

            visual_features = self.module.encode_visual(gt)
            visual_features = visual_features.reshape((visual_features.shape[0], -1))
        
        return {"y_pred": pred_features, "y": visual_features}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            phoneme, landmark, mask, ref, gt = batch
            
            (_, _, _, reconstructed_images) = self(
                phoneme = phoneme, 
                landmark = landmark,
                mask = mask,
                ref = ref
            )
            
            for i in range(reconstructed_images.shape[0]):
                img = (reconstructed_images[i] + 1) / 2
                save_image(img, f'{save_folder}/image_{i:05d}.jpg')
            