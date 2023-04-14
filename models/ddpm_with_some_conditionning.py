import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimePositionalEmbedding(nn.Module):
    def __init__(self, dimension, T=1000, device=None) -> None:
        super().__init__()
        self.embedding = torch.zeros(T, dimension)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2).float() * (-np.log(10000.0) / dimension))
        self.embedding[:, 0::2] = torch.sin(position * div_term)
        self.embedding[:, 1::2] = torch.cos(position * div_term)
        if device is not None:
            self.embedding = self.embedding.to(device)
    
    def forward(self, timestep):
        return self.embedding[timestep]
    
class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = self.weight.mean(dim=[1, 2, 3], keepdim=True)
        var = torch.var(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        normalized_weight = (self.weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8) -> None:
        super().__init__()
        self.conv = WeightStandardizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2) -> None:
        super().__init__()
        self.downsampler = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1)
    
    def forward(self, x):
        return self.downsampler(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2) -> None:
        super().__init__()
        self.upsampler = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=scale_factor, padding=1)
    
    def forward(self, x):
        return self.upsampler(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, cemb_dim, groups=8) -> None:
        super().__init__()
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, out_channels),
        ) if temb_dim is not None else nn.Identity()

        self.cemb_proj = nn.Sequential(
            nn.Linear(cemb_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        ) if cemb_dim is not None else nn.Identity()

        self.block_a = ConvBlock(in_channels, out_channels, groups=groups)
        self.block_b = ConvBlock(out_channels, out_channels, groups=groups)
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb, condition):
        h = self.block_a(x)
        h = self.cemb_proj(condition)[:, :, None, None] * h + self.temb_proj(temb)[:, :, None, None]
        h = self.block_b(h)
        return h + self.residual_proj(x)
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, head_dim=32, groups=32) -> None:
        super().__init__()
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q = nn.Conv2d(in_channels, num_heads * head_dim, kernel_size=1)
        self.k = nn.Conv2d(in_channels, num_heads * head_dim, kernel_size=1)
        self.v = nn.Conv2d(in_channels, num_heads * head_dim, kernel_size=1)
        self.norm = nn.GroupNorm(groups, in_channels)
        self.proj = nn.Conv2d(num_heads * head_dim, in_channels, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        q = self.q(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = self.k(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 2, 3)
        v = self.v(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attention = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        attention = torch.matmul(attention, v)
        attention = attention.permute(0, 1, 3, 2).contiguous().view(B, self.num_heads * self.head_dim, H, W)
        return self.norm(x + self.proj(attention))
    
class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, cemb_dim, downsample=True, attn=False, num_blocks=2, groups=8) -> None:
        super().__init__()

        self.resnet = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, temb_dim, cemb_dim, groups=groups)
            for i in range(num_blocks)
        ])

        self.attn = nn.ModuleList([
            SelfAttention(out_channels, num_heads=8, head_dim=32, groups=groups)
            if attn else nn.Identity()
            for _ in range(num_blocks)
        ])

        self.downsample = Downsample(out_channels, out_channels) if downsample else nn.Identity()

    def forward(self, x, temb, cemb):
        for resnet_block, attn_block in zip(self.resnet, self.attn):
            x = resnet_block(x, temb, cemb)
            x = attn_block(x)
        return self.downsample(x)
    
class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, cemb_dim, upsample=True, attn=False, num_blocks=2, groups=8) -> None:
        super().__init__()
        self.resnet = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, temb_dim, cemb_dim, groups=groups)
            for i in range(num_blocks)
        ])
        self.attn = nn.ModuleList([
            SelfAttention(out_channels, num_heads=8, head_dim=32, groups=groups)
            if attn else nn.Identity()
            for _ in range(num_blocks)
        ])
        self.upsample = Upsample(out_channels, out_channels) if upsample else nn.Identity()

    def forward(self, x, temb, cemb):
        for resnet_block, attn_block in zip(self.resnet, self.attn):
            x = resnet_block(x, temb, cemb)
            x = attn_block(x)
        return self.upsample(x)

class ResUNet(nn.Module):
    def __init__(self, in_channels, condition_dim, drop_prob=0.1, T=1000) -> None:
        super().__init__()
        self.drop_prob = 1 - drop_prob
        self.in_conv = nn.Conv2d(in_channels, 128, kernel_size=3, padding='same')
        self.positional_encoder = nn.Sequential(
            TimePositionalEmbedding(dimension=128, T=T, device='cuda'),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.condition_encoder = nn.Sequential(
            Downsample(condition_dim, 64, scale_factor=4),
            nn.ReLU(),
            Downsample(64, 128, scale_factor=2),
            nn.ReLU(),
            Downsample(128, 128, scale_factor=2),
            nn.ReLU(),
            Downsample(128, 128, scale_factor=2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

        self.encoder = nn.ModuleList([
            EncodingBlock(in_channels=128, out_channels=128, temb_dim=128 * 4, cemb_dim=128, downsample=True, attn=False, num_blocks=2, groups=32),
            EncodingBlock(in_channels=128, out_channels=256, temb_dim=128 * 4, cemb_dim=128, downsample=True, attn=False, num_blocks=2, groups=32),
            EncodingBlock(in_channels=256, out_channels=256, temb_dim=128 * 4, cemb_dim=128, downsample=True, attn=True, num_blocks=2, groups=32),
            EncodingBlock(in_channels=256, out_channels=512, temb_dim=128 * 4, cemb_dim=128, downsample=True, attn=False, num_blocks=2, groups=32)
        ])

        self.bottleneck = EncodingBlock(in_channels=512, out_channels=512, temb_dim=128 * 4, 
                                        cemb_dim=128,  downsample=False, attn=True, num_blocks=2, groups=32)

        self.decoder = nn.ModuleList([
            DecodingBlock(in_channels=512 + 512, out_channels=512, temb_dim=128 * 4, cemb_dim=128, upsample=True, attn=False, num_blocks=2, groups=32),
            DecodingBlock(in_channels=512 + 256, out_channels=256, temb_dim=128 * 4, cemb_dim=128, upsample=True, attn=True, num_blocks=2, groups=32),
            DecodingBlock(in_channels=256 + 256, out_channels=256, temb_dim=128 * 4, cemb_dim=128, upsample=True, attn=False, num_blocks=2, groups=32),
            DecodingBlock(in_channels=256 + 128, out_channels=128, temb_dim=128 * 4, cemb_dim=128, upsample=True, attn=False, num_blocks=2, groups=32)
        ])

        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, time, condition, context_free=False):
        assert x.shape[0] == time.shape[0], 'Batch size of x and time must be the same'
        temb = self.positional_encoder(time)

        if not context_free:
            context_mask = torch.bernoulli(torch.full(size=(condition.shape[0],), fill_value=self.drop_prob)).to(device)
            condition = condition * context_mask[:, None, None, None]
        else:
            condition = torch.zeros_like(condition, dtype=torch.float32, device=device)
        cemb = self.condition_encoder(condition)

        skip_connections = []

        x = self.in_conv(x)
        skip_connections.append(x)
        
        # encoding part
        for block in self.encoder:
            x = block(x, temb, cemb)
            skip_connections.append(x)

        # bottleneck
        x = self.bottleneck(x, temb, cemb)

        # decoding part
        for block in self.decoder:
            x = block(torch.cat([x, skip_connections.pop()], dim=1), temb, cemb)

        x = torch.cat([x, skip_connections.pop()], dim=1)
        assert len(skip_connections) == 0, 'Skip connections must be empty'
        return self.out_conv(x)

class DiffusionModule(nn.Module):
    def __init__(self, betas=(0.0001, 0.02), T=1000) -> None:
        super().__init__()
        self.T = T
        self.betas = torch.linspace(betas[0], betas[1], T)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0).to(device)

    def forward_process(self, images, time):
        noise = torch.randn_like(images).to(images.device)
        alpha_hat = self.alphas_hat[time, None, None, None]
        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * noise, noise
    
    def reverse_process(self, model, x_t, time, condition, context_free=False):
        return model(x_t, time, condition, context_free=context_free)
    
    @torch.no_grad()
    def sample(self, model, x_T, condition, w=0.):
        x_t = x_T
        for timestep in tqdm(range(self.T - 1, -1, -1), desc='Sampling', position=0, leave=True):
            times = torch.full(size=(x_T.shape[0],), fill_value=timestep, dtype=torch.long, device=x_t.device)
            
            # preds
            guided_prediction = self.reverse_process(model, x_t, times, condition, context_free=False)
            free_prediction = self.reverse_process(model, x_t, times, condition, context_free=True)
            predicted = w * guided_prediction + (1 - w) * free_prediction

            # diffusions
            beta_t = self.betas[timestep].to(x_t.device)
            alpha_t = self.alphas[timestep].to(x_t.device)
            alpha_hat_t = self.alphas_hat[timestep].to(x_t.device)
            alpha_hat_t_prev = self.alphas_hat[timestep - 1].to(x_t.device)
            beta_t_hat = (1 - alpha_hat_t_prev) / (1 - alpha_hat_t) * beta_t
            if timestep > 0:
                var = torch.sqrt(beta_t_hat) * torch.randn_like(x_t).to(x_t.device)
            else :
                var = 0
            x_t = alpha_t.rsqrt() * (x_t - beta_t / torch.sqrt((1 - alpha_hat_t_prev)) * predicted) + var
        return x_t
