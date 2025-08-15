import torch
import torch.nn.functional as F
import numpy as np

class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.mean_noise_coeff = (1. - self.alphas) / self.sqrt_one_minus_alphas_cumprod
        self.posterior_variance = (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) * self.betas
        self.posterior_variance[0] = self.posterior_variance[1]
        self.posterior_std_dev = torch.sqrt(self.posterior_variance + 1e-8)  # 개선: epsilon for stability

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1 + 1e-8)  # 개선: epsilon

    def add_noise(self, original, t):
        t = t.long()  # 개선: 타입 안전
        noise = torch.randn_like(original)
        noised_data = self.sqrt_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1) * original + \
                      self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1) * noise
        return noised_data, noise

    def p_step(self, model_output, t, sample):
        t = t.long()  # 개선: 타입 안전
        coef_sample = self.sqrt_recip_alphas.gather(-1, t).reshape(-1, 1, 1)
        coef_noise = self.mean_noise_coeff.gather(-1, t).reshape(-1, 1, 1)
        std_dev = self.posterior_std_dev.gather(-1, t).reshape(-1, 1, 1)
        mean = coef_sample * (sample - coef_noise * model_output)
        
        noise = torch.randn_like(sample)
        mask = (t != 0).float().reshape(-1, 1, 1)
        
        return mean + mask * std_dev * noise
    
    def predict_x0_from_noise(self, x_t, t, noise):
        t = t.long()  # 개선: 타입 안전
        return self.sqrt_recip_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1) * x_t - \
               self.sqrt_recipm1_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1) * noise