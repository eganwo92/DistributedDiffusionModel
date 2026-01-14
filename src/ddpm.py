import torch
import torch.nn.functional as F
import numpy as np

class DDPM:
    """Denoising Diffusion Probabilistic Model."""
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Precompute for q_sample
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Precompute for q_posterior
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, t, y, clip_denoised=True):
        """Sample from p(x_{t-1} | x_t)."""
        # Predict noise
        pred_noise = model(x, t, y)

        # Compute x_0
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        pred_x_start = sqrt_recip_alphas_t * (x - sqrt_one_minus_alphas_cumprod_t * pred_noise)

        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)

        # Compute mean
        posterior_mean_coef1 = (self.betas[t] * torch.sqrt(self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t])).reshape(-1, 1, 1, 1)
        posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev[t]) * torch.sqrt(self.alphas[t]) / (1.0 - self.alphas_cumprod[t])).reshape(-1, 1, 1, 1)
        posterior_mean = posterior_mean_coef1 * pred_x_start + posterior_mean_coef2 * x

        # Sample
        # During sampling, all timesteps in batch are the same
        if t[0] == 0:
            return posterior_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, n, img_size, y, seed=None, verbose=False):
        """Generate samples using the reverse diffusion process."""
        if seed is not None:
            torch.manual_seed(seed)

        model.eval()
        device = next(model.parameters()).device
        shape = (n, 3, img_size, img_size)

        # Start from pure noise
        img = torch.randn(shape, device=device)

        # Reverse diffusion
        indices = list(range(self.timesteps))[::-1]
        if verbose:
            from tqdm import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = torch.full((n,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, y)

        return img
