from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .base import (
    TimePositionalEmbedding,
    EncodingBlock,
    DecodingBlock
)

class DDPM(pl.LightningModule):
    def __init__(
        self, 
        diffusion,
        sampler,
        in_channels, 
        out_channels,
        image_size,
        timesteps    = 1000,
        lr           = 1e-5,
        weight_decay = 1e-7,
        precision    = 32,
        **kwargs
    ) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.sampler = sampler
        self.precision = torch.float16 if precision == 16 else torch.float32

        # achitecture modules
        self.in_conv = nn.Conv2d(in_channels, 128, kernel_size=3, padding='same')
        self.positional_encoder = nn.Sequential(
            TimePositionalEmbedding(dimension=128, T=timesteps),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.encoder = nn.ModuleList([
            EncodingBlock(in_channels=128, out_channels=128, temb_dim=128 * 4, downsample=True, attn=False, num_blocks=2, groups=32),
            EncodingBlock(in_channels=128, out_channels=256, temb_dim=128 * 4, downsample=True, attn=False, num_blocks=2, groups=32),
            EncodingBlock(in_channels=256, out_channels=256, temb_dim=128 * 4, downsample=True, attn=True, num_blocks=2, groups=32),
            EncodingBlock(in_channels=256, out_channels=512, temb_dim=128 * 4, downsample=True, attn=False, num_blocks=2, groups=32)
        ])

        self.bottleneck = EncodingBlock(in_channels=512, out_channels=512, temb_dim=128 * 4, downsample=False, attn=True, num_blocks=2, groups=32)

        self.decoder = nn.ModuleList([
            DecodingBlock(in_channels=512 + 512, out_channels=512, temb_dim=128 * 4, upsample=True, attn=False, num_blocks=2, groups=32),
            DecodingBlock(in_channels=512 + 256, out_channels=256, temb_dim=128 * 4, upsample=True, attn=True, num_blocks=2, groups=32),
            DecodingBlock(in_channels=256 + 256, out_channels=256, temb_dim=128 * 4, upsample=True, attn=False, num_blocks=2, groups=32),
            DecodingBlock(in_channels=256 + 128, out_channels=128, temb_dim=128 * 4, upsample=True, attn=False, num_blocks=2, groups=32)
        ])

        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1)
        )

        self.save_hyperparameters(ignore=['diffusion', 'sampler'])

    def on_train_start(self) -> None:
        self.positional_encoder[0] = self.positional_encoder[0].to(self.device)
        self.diffusion = self.diffusion.to(self.device)

    def forward(self, x, time):
        assert x.shape[0] == time.shape[0], 'Batch size of x and time must be the same'
        temb = self.positional_encoder(time)
        skip_connections = []

        x = self.in_conv(x)
        skip_connections.append(x)
        
        # encoding part
        for block in self.encoder:
            x = block(x, temb)
            skip_connections.append(x)

        # bottleneck
        x = self.bottleneck(x, temb)

        # decoding part
        for block in self.decoder:
            x = block(torch.cat([x, skip_connections.pop()], dim=1), temb)

        x = torch.cat([x, skip_connections.pop()], dim=1)
        assert len(skip_connections) == 0, 'Skip connections must be empty'
        return self.out_conv(x)
    
    def training_step(self, batch, batch_idx):
        x_0 = batch[0]
        x_0 = x_0.to(self.precision)

        times, _ = self.sampler.sample(device=self.device)

        x_t, noise = self.diffusion.forward_process(x_0, times)
        x_t = x_t.to(self.precision)

        noise_hat = self.diffusion.reverse_process(self, x_t, times)

        loss = F.mse_loss(noise_hat, noise)

        self.log('mse_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay, 
            betas=(0.5, 0.9)
        )
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_steps, eta_min=1e-9, last_epoch=-1
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def sample_img(
        self, 
        n_samples   = 1, 
        noise       = None
    ):
        if noise is None:
            noise = torch.randn(
                n_samples, 
                self.hparams.in_channels,
                self.hparams.image_size[0], 
                self.hparams.image_size[1]
            ).to(self.device, dtype=torch.float32)
        
        samples = self.diffusion.sample(self, noise, condition=None)

        return samples

