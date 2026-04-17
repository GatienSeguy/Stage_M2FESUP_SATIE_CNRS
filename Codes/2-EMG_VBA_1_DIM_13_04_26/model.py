import torch
import torch.nn as nn
import numpy as np


def get_time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) * (np.log(10000.0) / half))
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class UNet(nn.Module):
    """
    UNet pour images 64x64 RGB.
    3 etages de downsampling : 64 -> 32 -> 16 -> 8 (bottleneck)
    3 etages de upsampling   : 8 -> 16 -> 32 -> 64
    """
    
    def __init__(self, in_ch=3, base_ch=128, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        ch1, ch2, ch3, ch_mid = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        
        # === TIME EMBEDDING ===
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # === ENCODER ===
        
        # Bloc 1 : (B, 3, 64, 64) -> pool -> (B, ch1, 32, 32)
        self.conv1a = nn.Conv2d(in_ch, ch1, 3, padding=1)
        self.conv1b = nn.Conv2d(ch1, ch1, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.time_proj1 = nn.Linear(time_dim, ch1)
        self.norm1a = nn.GroupNorm(8, ch1)
        self.norm1b = nn.GroupNorm(8, ch1)
        
        # Bloc 2 : (B, ch1, 32, 32) -> pool -> (B, ch2, 16, 16)
        self.conv2a = nn.Conv2d(ch1, ch2, 3, padding=1)
        self.conv2b = nn.Conv2d(ch2, ch2, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.time_proj2 = nn.Linear(time_dim, ch2)
        self.norm2a = nn.GroupNorm(8, ch2)
        self.norm2b = nn.GroupNorm(8, ch2)
        
        # Bloc 3 : (B, ch2, 16, 16) -> pool -> (B, ch3, 8, 8)
        self.conv3a = nn.Conv2d(ch2, ch3, 3, padding=1)
        self.conv3b = nn.Conv2d(ch3, ch3, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.time_proj3 = nn.Linear(time_dim, ch3)
        self.norm3a = nn.GroupNorm(8, ch3)
        self.norm3b = nn.GroupNorm(8, ch3)
            
        # === BOTTLENECK === (B, ch3, 8, 8) -> (B, ch_mid, 8, 8) -> (B, ch3, 8, 8)
        self.conv_mid1 = nn.Conv2d(ch3, ch_mid, 3, padding=1)
        self.conv_mid2 = nn.Conv2d(ch_mid, ch3, 3, padding=1)
        self.time_proj_mid = nn.Linear(time_dim, ch_mid)
        self.norm_mid1 = nn.GroupNorm(8, ch_mid)
        self.norm_mid2 = nn.GroupNorm(8, ch3)
        
        # === DECODER ===
        
        # Up Bloc 3 : (B, ch3, 8, 8) -> up -> cat skip3 -> (B, ch2, 16, 16)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3a = nn.Conv2d(ch3 + ch3, ch2, 3, padding=1)
        self.conv_up3b = nn.Conv2d(ch2, ch2, 3, padding=1)
        self.time_proj_up3 = nn.Linear(time_dim, ch2)
        self.norm_up3a = nn.GroupNorm(8, ch2)
        self.norm_up3b = nn.GroupNorm(8, ch2)
        
        # Up Bloc 2 : (B, ch2, 16, 16) -> up -> cat skip2 -> (B, ch1, 32, 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2a = nn.Conv2d(ch2 + ch2, ch1, 3, padding=1)
        self.conv_up2b = nn.Conv2d(ch1, ch1, 3, padding=1)
        self.time_proj_up2 = nn.Linear(time_dim, ch1)
        self.norm_up2a = nn.GroupNorm(8, ch1)
        self.norm_up2b = nn.GroupNorm(8, ch1)
        
        # Up Bloc 1 : (B, ch1, 32, 32) -> up -> cat skip1 -> (B, ch1, 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1a = nn.Conv2d(ch1 + ch1, ch1, 3, padding=1)
        self.conv_up1b = nn.Conv2d(ch1, ch1, 3, padding=1)
        self.time_proj_up1 = nn.Linear(time_dim, ch1)
        self.norm_up1a = nn.GroupNorm(8, ch1)
        self.norm_up1b = nn.GroupNorm(8, ch1)
        
        # === SORTIE ===
        self.out = nn.Conv2d(ch1, in_ch, 1)
        
        self.act = nn.SiLU()
    
    def forward(self, x, t):
        t_emb = get_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # === ENCODER ===
        h = self.act(self.norm1a(self.conv1a(x)))
        h = h + self.time_proj1(t_emb)[:, :, None, None]
        h = self.act(self.norm1b(self.conv1b(h)))
        skip1 = h
        h = self.pool1(h)
        
        h = self.act(self.norm2a(self.conv2a(h)))
        h = h + self.time_proj2(t_emb)[:, :, None, None]
        h = self.act(self.norm2b(self.conv2b(h)))
        skip2 = h
        h = self.pool2(h)
        
        h = self.act(self.norm3a(self.conv3a(h)))
        h = h + self.time_proj3(t_emb)[:, :, None, None]
        h = self.act(self.norm3b(self.conv3b(h)))
        skip3 = h
        h = self.pool3(h)
        
        # === BOTTLENECK ===
        h = self.act(self.norm_mid1(self.conv_mid1(h)))
        h = h + self.time_proj_mid(t_emb)[:, :, None, None]
        h = self.act(self.norm_mid2(self.conv_mid2(h)))
        
        # === DECODER ===
        h = self.up3(h)
        h = torch.cat([h, skip3], dim=1)
        h = self.act(self.norm_up3a(self.conv_up3a(h)))
        h = h + self.time_proj_up3(t_emb)[:, :, None, None]
        h = self.act(self.norm_up3b(self.conv_up3b(h)))
        
        h = self.up2(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.act(self.norm_up2a(self.conv_up2a(h)))
        h = h + self.time_proj_up2(t_emb)[:, :, None, None]
        h = self.act(self.norm_up2b(self.conv_up2b(h)))
        
        h = self.up1(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.act(self.norm_up1a(self.conv_up1a(h)))
        h = h + self.time_proj_up1(t_emb)[:, :, None, None]
        h = self.act(self.norm_up1b(self.conv_up1b(h)))
        
        # === SORTIE ===
        return self.out(h)