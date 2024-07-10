import torch
import torch.nn as nn
from .attention import default, exists
from torch import nn, einsum
from einops import rearrange, repeat

class VisualEncoder(nn.Module):
    def __init__(self, attention_head_dim=512, context_dim=512, num_classes=100):
        super(VisualEncoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        )
        self.ssa1 = CrossAttention(query_dim=64)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.ssa2 = CrossAttention(query_dim=128)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.ssa3 = CrossAttention(query_dim=256)
       
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )        
        self.ssa4 = CrossAttention(query_dim=512)
        
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )
            
    def cross_attn(self, model, x, context=None):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        context = rearrange(context, 'b c h w -> b (h w) c')
        x, attn = model(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        attn = rearrange(attn, '(b h) d c -> b d (h c)', h=8)
        return x, attn
    
    def forward(self, x, audio):    
        (a1,a2,a3,a4) = audio
        x = self.conv1(x)
        x1, attn1 = self.cross_attn(self.ssa1, x, a1)
        x = self.conv2(x1)
        x2, attn2 = self.cross_attn(self.ssa2, x, a2)
        x = self.conv3(x2)
        x3, attn3 = self.cross_attn(self.ssa3, x, a3)
        x = self.conv4(x3)
        x4, attn4 = self.cross_attn(self.ssa4, x, a4)
        x = self.final(x4)
            
        return x, (x1, x2, x3, x4)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        return self.to_out(out), attn