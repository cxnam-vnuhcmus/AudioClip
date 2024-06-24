
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import re
from g2p_en import G2p
import numpy as np
import os

class PhonemeEncoder(nn.Module):
    def __init__(self, dim_out=128):
        super().__init__()
        self.g2p = G2p()
        self.fc = nn.Linear(256, dim_out)
        
    def tokenizer(self, phonemes):
        with torch.no_grad():
            tokens = []
            for phoneme in phonemes:
                phoneme = re.sub(r'\d+$', '', phoneme)
                enc = self.g2p.encode(phoneme)
                enc = self.g2p.gru(enc, len(phoneme) + 1, self.g2p.enc_w_ih, self.g2p.enc_w_hh,
                                self.g2p.enc_b_ih, self.g2p.enc_b_hh, h0=np.zeros((1, self.g2p.enc_w_hh.shape[-1]), np.float32))
                last_hidden = enc[:, -1, :]
                tokens.append(last_hidden)
            tokens = [torch.tensor(arr) for arr in tokens]
            tokens = torch.cat(tokens, axis=0)
        return tokens
 
    def forward(self, tokens):
        tokens = self.fc(tokens)
        return tokens
