import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps, dim):
    # print(f"get_timestep_embedding 输入: timesteps={timesteps.shape}, dim={dim}")
    
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    
    # print(f"get_timestep_embedding 输出: {emb.shape}")
    return emb

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        
    def forward(self, x):
        return x + self.net(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        h = self.heads
        
        # 确保输入维度正确
        q = self.to_q(x)  # [B, query_dim] -> [B, inner_dim]
        k = self.to_k(context)  # [B, context_dim] -> [B, inner_dim]
        v = self.to_v(context)  # [B, context_dim] -> [B, inner_dim]
        
        # 重塑为多头格式 [B, h, d]
        q = q.view(q.shape[0], h, -1)  # [B, inner_dim] -> [B, h, inner_dim//h]
        k = k.view(k.shape[0], h, -1)
        v = v.view(v.shape[0], h, -1)
        
        # 注意力计算
        sim = torch.einsum('b h d, b h d -> b h', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        # 应用注意力权重
        out = torch.einsum('b h, b h d -> b h d', attn, v)
        
        # 重塑回原始格式
        out = out.reshape(out.shape[0], -1)  # [B, h, d] -> [B, inner_dim]
        return self.to_out(out)

class ImprovedDiffusionRegressor(nn.Module):
    def __init__(self, target_dim, cond_dim, denoiser_hidden=512, timesteps=1000,
                 beta_start=1e-4, beta_end=2e-2, t_emb_dim=256, num_res_blocks=4,
                 dropout=0.1, use_attention=True):
        super().__init__()
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.t_emb_dim = t_emb_dim
        self.use_attention = use_attention

        # Enhanced noise schedule
        betas = cosine_beta_schedule(timesteps, s=0.01)  # Adjusted s parameter
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Enhanced timestep embedding
        self.t_emb_proj = nn.Sequential(
            nn.Linear(t_emb_dim, denoiser_hidden),
            nn.GELU(),
            nn.Linear(denoiser_hidden, denoiser_hidden),
            nn.GELU(),
            nn.Linear(denoiser_hidden, t_emb_dim),
        )

        # Enhanced denoiser architecture
        self.input_proj = nn.Linear(target_dim + t_emb_dim + cond_dim, denoiser_hidden)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(denoiser_hidden, dropout=dropout) 
            for _ in range(num_res_blocks)
        ])
        
        # Cross-attention for conditioning
        if use_attention:
            self.cross_attn = CrossAttention(denoiser_hidden, cond_dim, heads=8, dim_head=64, dropout=dropout)
            self.attn_norm = nn.LayerNorm(denoiser_hidden)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(denoiser_hidden, denoiser_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(denoiser_hidden // 2, target_dim),
        )

        # Enhanced direct regressor with constraint-aware output
        self.direct_head = nn.Sequential(
            nn.Linear(cond_dim, denoiser_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(denoiser_hidden // 2, target_dim),
            nn.ReLU()  # Constrain outputs to sum to 1 for abundances
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def q_sample(self, y0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_alpha_bar * y0 + sqrt_one_minus * noise

    def predict_noise(self, y_t, t, cond):
        t_emb = get_timestep_embedding(t, self.t_emb_dim)
        t_emb = self.t_emb_proj(t_emb)
        
        # Project input
        x = torch.cat([y_t, t_emb, cond], dim=1)
        x = self.input_proj(x)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply cross-attention if enabled
        if self.use_attention:
            attn_out = self.cross_attn(x, cond)
            x = self.attn_norm(x + attn_out)
        
        # Final output
        eps_pred = self.output_layers(x)
        return eps_pred

    def forward(self, y0, cond):
        B = y0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=y0.device, dtype=torch.long)
        noise = torch.randn_like(y0)
        y_t = self.q_sample(y0, t, noise=noise)
        eps_pred = self.predict_noise(y_t, t, cond)
        
        return {
            'y_t': y_t, 
            't': t, 
            'noise': noise, 
            'eps_pred': eps_pred,
            'direct_pred': self.direct_head(cond)  # Add direct prediction for multi-task learning
        }

    def predict_x0_from_y_t(self, y_t, t, cond):
        eps_pred = self.predict_noise(y_t, t, cond)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_bar_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        x0_pred = (y_t - sqrt_one_minus_bar_t * eps_pred) / (sqrt_alpha_bar_t + 1e-8)
        return x0_pred

    @torch.no_grad()
    def p_sample(self, y_t, t, cond, guidance_scale=1.5):
        # Classifier-free guidance for better conditioning
        if guidance_scale != 1.0:
            # Zero conditioning for unconditional prediction
            uncond_cond = torch.zeros_like(cond)
            eps_uncond = self.predict_noise(y_t, t, uncond_cond)
            eps_cond = self.predict_noise(y_t, t, cond)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
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
    def sample_from_condition(self, cond, num_steps=None, guidance_scale=1.5):
        B = cond.shape[0]
        if num_steps is None:
            steps = list(range(self.timesteps - 1, -1, -1))
        else:
            step_size = self.timesteps // num_steps
            steps = list(range(self.timesteps - 1, -1, -step_size))
            
        y_t = torch.randn((B, self.target_dim), device=cond.device)
        for t_int in steps:
            t = torch.full((B,), t_int, device=cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, cond, guidance_scale=guidance_scale)
            
        # Apply softmax to ensure valid abundance distribution
        y_t = torch.relu(y_t)
        return y_t