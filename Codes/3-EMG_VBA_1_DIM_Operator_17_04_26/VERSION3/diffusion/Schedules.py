import torch
import numpy as np


class NoiseSchedule:
    pass


class DDPMSchedule(NoiseSchedule):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.variant = 'ddpm'
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar = alpha_bar
        self.sigmas = torch.sqrt(1.0 - alpha_bar)


class VESchedule(NoiseSchedule):
    def __init__(self, T=1000, sigma_min=0.01, sigma_max=50.0):
        self.T = T
        self.variant = 've'
        self.sigmas = torch.exp(
            torch.linspace(np.log(sigma_min), np.log(sigma_max), T)
        )
        self.alphas_bar = torch.ones(T)
        self.betas = None
        self.alphas = None


class VPOUSchedule(NoiseSchedule):
    """VP Ornstein–Uhlenbeck : beta(t)"""
    def __init__(self, T=1000, beta_min=0.1, beta_max=20.0):
        self.T = T
        self.variant = 'vpou'
        self.betas = (beta_min + (beta_max - beta_min) * torch.arange(T) / max(T - 1, 1)) / T
        self.betas = self.betas.clamp(max=0.999)
        self.alphas = 1.0 - self.betas
        alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar = alpha_bar
        self.sigmas = torch.sqrt(1.0 - alpha_bar)