import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_classes, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.class_mlp = nn.Embedding(num_classes, out_ch)
        
        # GroupNorm: use min(8, in_ch) groups to ensure divisibility
        groups1 = min(8, in_ch)
        groups2 = min(8, out_ch)
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups1, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups2, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb, y):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        class_emb = self.class_mlp(y)
        h = h + (time_emb + class_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class TinyUNet(nn.Module):
    """A small U-Net for conditional diffusion."""
    def __init__(self, num_classes=10, base_ch=96, time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embed = SinusoidalPositionalEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Down
        self.down1 = ResidualBlock(3, base_ch, time_emb_dim, num_classes)
        self.down2 = ResidualBlock(base_ch, base_ch * 2, time_emb_dim, num_classes)
        self.down3 = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim, num_classes)
        self.downsample = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)

        # Middle
        self.mid1 = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim, num_classes)
        self.mid2 = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim, num_classes)

        # Up
        self.upsample = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.up1 = ResidualBlock(base_ch * 4, base_ch * 2, time_emb_dim, num_classes)
        self.up2 = ResidualBlock(base_ch * 4, base_ch, time_emb_dim, num_classes)
        self.up3 = ResidualBlock(base_ch * 2, base_ch, time_emb_dim, num_classes)
        # GroupNorm: use min(8, base_ch) groups
        out_groups = min(8, base_ch)
        self.out = nn.Sequential(
            nn.GroupNorm(out_groups, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 3, 3, padding=1),
        )

    def forward(self, x, t, y):
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        # Down
        h1 = self.down1(x, t_emb, y)
        h2 = self.down2(h1, t_emb, y)
        h3 = self.down3(h2, t_emb, y)
        h = self.downsample(h3)

        # Middle
        h = self.mid1(h, t_emb, y)
        h = self.mid2(h, t_emb, y)

        # Up
        h = self.upsample(h)
        h = torch.cat([h, h3], dim=1)
        h = self.up1(h, t_emb, y)
        h = torch.cat([h, h2], dim=1)
        h = self.up2(h, t_emb, y)
        h = torch.cat([h, h1], dim=1)
        h = self.up3(h, t_emb, y)
        return self.out(h)
