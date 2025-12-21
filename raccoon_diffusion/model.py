"""
Tiny U-Net for 64x64 diffusion model.
Optimized for training on Apple Silicon (M4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings using sinusoidal encoding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Basic conv block with group norm and SiLU."""

    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        else:
            self.time_mlp = None

        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Simple self-attention block."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


class TinyUNet(nn.Module):
    """
    Tiny U-Net for 64x64 images.

    Architecture:
    - 4 resolution levels: 64 -> 32 -> 16 -> 8
    - Channel progression: 64 -> 128 -> 256 -> 256
    - Attention at 16x16 and 8x8 resolutions
    - ~2.5M parameters (trainable on M4 in reasonable time)
    """

    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)

        # Encoder (downsampling path)
        self.down1 = ConvBlock(64, 64, time_emb_dim)      # 64x64
        self.down2 = ConvBlock(64, 128, time_emb_dim)     # 32x32
        self.down3 = ConvBlock(128, 256, time_emb_dim)    # 16x16
        self.attn1 = AttentionBlock(256)
        self.down4 = ConvBlock(256, 256, time_emb_dim)    # 8x8
        self.attn2 = AttentionBlock(256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 256, time_emb_dim)
        self.bottleneck_attn = AttentionBlock(256)

        # Decoder (upsampling path)
        self.up4 = ConvBlock(512, 256, time_emb_dim)      # 8x8
        self.up_attn2 = AttentionBlock(256)
        self.up3 = ConvBlock(512, 128, time_emb_dim)      # 16x16
        self.up_attn1 = AttentionBlock(128)
        self.up2 = ConvBlock(256, 64, time_emb_dim)       # 32x32
        self.up1 = ConvBlock(128, 64, time_emb_dim)       # 64x64

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial
        x = self.init_conv(x)

        # Encoder
        d1 = self.down1(x, t_emb)           # 64x64, 64ch
        d2 = self.down2(self.pool(d1), t_emb)  # 32x32, 128ch
        d3 = self.down3(self.pool(d2), t_emb)  # 16x16, 256ch
        d3 = self.attn1(d3)
        d4 = self.down4(self.pool(d3), t_emb)  # 8x8, 256ch
        d4 = self.attn2(d4)

        # Bottleneck
        b = self.bottleneck(self.pool(d4), t_emb)  # 4x4, 256ch
        b = self.bottleneck_attn(b)

        # Decoder
        u4 = F.interpolate(b, scale_factor=2, mode='nearest')
        u4 = self.up4(torch.cat([u4, d4], dim=1), t_emb)
        u4 = self.up_attn2(u4)

        u3 = F.interpolate(u4, scale_factor=2, mode='nearest')
        u3 = self.up3(torch.cat([u3, d3], dim=1), t_emb)
        u3 = self.up_attn1(u3)

        u2 = F.interpolate(u3, scale_factor=2, mode='nearest')
        u2 = self.up2(torch.cat([u2, d2], dim=1), t_emb)

        u1 = F.interpolate(u2, scale_factor=2, mode='nearest')
        u1 = self.up1(torch.cat([u1, d1], dim=1), t_emb)

        return self.out_conv(u1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = TinyUNet()
    print(f"Model parameters: {count_parameters(model):,}")

    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,)).float()

    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
