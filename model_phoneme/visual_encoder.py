import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")

    def encode(self, x):
        with torch.no_grad():
            latents = self.vae.encode(x).latent_dist.sample()
        return latents
    
    def decode(self, latents):
        reconstructed = self.vae.decode(latents)
        return reconstructed.sample
    
    def forward(self, x):
        with torch.no_grad():
            latents = self.vae.encode(x).latent_dist.sample()

        reconstructed = self.vae.decode(latents)
        return reconstructed.sample