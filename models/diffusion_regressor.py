
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb  # (B, dim)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()



class DiffusionRegressor(nn.Module):
    def __init__(self, target_dim, cond_dim, denoiser_hidden=512, timesteps=1000,
                 beta_start=1e-4, beta_end=2e-2, t_emb_dim=128):
        super().__init__()
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.t_emb_dim = t_emb_dim

        # noise schedule buffers
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # timestep embedding projection
        self.t_emb_proj = nn.Sequential(
            nn.Linear(t_emb_dim, denoiser_hidden),
            nn.ReLU(),
            nn.Linear(denoiser_hidden, t_emb_dim),
        )

        # denoiser MLP: input = [y_t | t_emb | cond]
        in_dim = target_dim + t_emb_dim + cond_dim
        self.denoiser = nn.Sequential(
            nn.Linear(in_dim, denoiser_hidden),
            nn.ReLU(),
            nn.Linear(denoiser_hidden, denoiser_hidden),
            nn.ReLU(),
            nn.Linear(denoiser_hidden, target_dim),
        )

        # optional direct regressor (fast inference baseline)
        self.direct_head = nn.Linear(cond_dim, target_dim)

    def q_sample(self, y0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_alpha_bar * y0 + sqrt_one_minus * noise

    def predict_noise(self, y_t, t, cond):
        t_emb = get_timestep_embedding(t, self.t_emb_dim)  # (B, t_emb_dim)
        t_emb = self.t_emb_proj(t_emb)
        x = torch.cat([y_t, t_emb, cond], dim=1)

        # print(x.shape)
        # print(y_t.shape, " ", t_emb.shape, " ", cond.shape)

        eps_pred = self.denoiser(x)
        return eps_pred

    def forward(self, y0, cond):
        """
        Training forward:
        y0: (B, target_dim) ground truth abundance (normalized)
        cond: (B, cond_dim) proj_embed
        returns: dict with y_t, t, noise, eps_pred
        """

        B = y0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=y0.device, dtype=torch.long)
        noise = torch.randn_like(y0)
        y_t = self.q_sample(y0, t, noise=noise)
        eps_pred = self.predict_noise(y_t, t, cond)
        return {'y_t': y_t, 't': t, 'noise': noise, 'eps_pred': eps_pred}

    def predict_x0_from_y_t(self, y_t, t, cond):
        eps_pred = self.predict_noise(y_t, t, cond)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        x0_pred = (y_t - sqrt_one_minus_bar_t * eps_pred) / (sqrt_alpha_bar_t + 1e-8)
        return x0_pred

    @torch.no_grad()
    def p_sample(self, y_t, t, cond):
        eps_pred = self.predict_noise(y_t, t, cond)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        x0_pred = (y_t - sqrt_one_minus_bar_t * eps_pred) / (sqrt_alpha_bar_t + 1e-8)

        beta_t = self.betas[t].unsqueeze(1)
        alpha_t = self.alphas[t].unsqueeze(1)
        alpha_bar_prev = self.alphas_cumprod_prev[t].unsqueeze(1)

        coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - self.alphas_cumprod[t].unsqueeze(1))
        coef2 = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - self.alphas_cumprod[t].unsqueeze(1))
        mean = coef1 * x0_pred + coef2 * y_t
        noise = torch.randn_like(y_t) if (t > 0).any() else torch.zeros_like(y_t)
        return mean + torch.sqrt(beta_t) * noise

    @torch.no_grad()
    def sample_from_condition(self, cond, num_steps=None):
        B = cond.shape[0]
        if num_steps is None:
            steps = list(range(self.timesteps - 1, -1, -1))
        else:
            steps = list(range(self.timesteps - 1, -1, -1))
        y_t = torch.randn((B, self.target_dim), device=cond.device)
        for t_int in steps:
            t = torch.full((B,), t_int, device=cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, cond)
        return y_t
