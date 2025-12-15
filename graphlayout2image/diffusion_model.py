import torch
import torch.nn as nn 
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, time_dim: int = 32, cond_dim: int = 64):
        super().__init__()

        # Time Embedding 
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), 
            nn.SiLU(), 
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)

        # Conditioning Cross-Attention 
        self.cross_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)

        # Decoder 
        self.dec3 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2)
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2)
        self.dec1 = nn.Conv2d(32, in_channels, 3, padding=1)
    
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        t_emb = self.time_mlp(t)
        
        # Encoder
        x1 = F.silu(self.enc1(x))
        x2 = F.silu(self.enc2(x1))
        x3 = F.silu(self.enc3(x2))

        # Reshape for attention 
        B, C, H, W = x3.shape
        x3_flat = x3.reshape(B, C, H*W).transpose(1, 2)
        x3_attn, _ = self.cross_attn(x3_flat, cond, cond)
        x3 = x3_attn.transpose(1, 2).reshape(B, C, H, W)

        # Decoder
        x = F.silu(self.dec3(x3))
        x = F.silu(self.dec2(x + x2))
        x = self.dec1(x + x1)

        return x


class SimpleDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def forward_diffusion(self, x0, t):
        """Add noise to images"""
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt()
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
