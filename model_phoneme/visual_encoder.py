import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class VisualEncoder(nn.Module):
    def __init__(self, autoencoder_model: AutoencoderKL):
        super(VisualEncoder, self).__init__()
        self.autoencoder = autoencoder_model

        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False
        for param in self.autoencoder.mid_block.parameters():
            param.requires_grad = False
        for param in self.autoencoder.decoder.parameters():
            param.requires_grad = True

    def encode(self, x):
        with torch.no_grad():
            latents = self.autoencoder.encode(x).latent_dist.sample()
        return latents
    
    def decode(self, latents):
        reconstructed = self.autoencoder.decode(latents).sample()
        return reconstructed
    
    def forward(self, x):
        with torch.no_grad():
            latents = self.autoencoder.encode(x).latent_dist.sample()

        reconstructed = self.autoencoder.decode(latents).sample()
        return reconstructed