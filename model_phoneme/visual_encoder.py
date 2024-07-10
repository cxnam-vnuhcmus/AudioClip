import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")

    def encode(self, x):
        with torch.no_grad():
            latents = self.vae.encode(x).latent_dist.mean
            latents = latents * 0.18215
        return latents
    
    def decode(self, latents):
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            reconstructed = self.vae.decode(latents).sample
            reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
        return reconstructed
    
    def forward(self, x):
        latents = self.encode(x)    
        reconstructed = self.decode(latents)
        return reconstructed