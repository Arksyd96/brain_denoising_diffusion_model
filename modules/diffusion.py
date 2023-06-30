import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class DiffusionModule(nn.Module):
    def __init__(self, T=1000, beta_schedule='cosine', input_perturbation=0.1) -> None:
        super().__init__()
        assert beta_schedule in ['linear', 'cosine'], 'beta_schedule must be either linear or cosine'
        self.T = T
        self.betas = self.cosine_beta_schedule(T) if beta_schedule == 'cosine' else self.linear_beta_schedule(T)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.input_perturbation = input_perturbation

    def to(self, device):
        self.alphas_hat = self.alphas_hat.to(device)
        return self

    def forward_process(self, images, time):
        noise = torch.randn_like(images).to(images.device)
        gamma = noise + self.input_perturbation * torch.randn_like(images).to(images.device)
        alpha_hat = self.alphas_hat[time, None, None, None]
        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gamma, noise
    
    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

    def cosine_beta_schedule(self, timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def reverse_process(self, model, x_t, time):#, condition, fill_value=0.1):
        # context_mask = torch.bernoulli(torch.full(size=(condition.shape[0],), fill_value=fill_value)).to(condition.device)
        # condition = condition * context_mask[:, None, None, None]
        # return model(torch.cat([x_t, condition], dim=1), time)
        return model(x_t, time)
    
    @torch.no_grad()
    def sample(self, model, x_T, condition, w=0.5):
        x_t = x_T
        for timestep in tqdm(range(self.T - 1, -1, -1), desc='Sampling', position=0, leave=True):
            times = torch.full(size=(x_T.shape[0],), fill_value=timestep, dtype=torch.long, device=x_t.device)

            # predictions
            guided_pred = self.reverse_process(model, x_t, times)#, condition, fill_value=1.)
            # free_pred = self.reverse_process(model, x_t, times, condition, fill_value=0.)
            # predicted = w * guided_pred + (1 - w) * free_pred
            predicted = guided_pred

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
