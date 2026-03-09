import torch, torch.nn as nn
from typing import Dict
class InformationBottleneckFusion(nn.Module):
    def __init__(self, d_in: int, d_latent: int = 128, beta: float = 0.1):
        super().__init__(); self.mu = nn.Linear(d_in, d_latent); self.logvar = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in); self.beta = float(beta)
    def set_ib_beta(self, beta: float): self.beta = float(beta)
    def forward(self, v_feat: torch.Tensor, a_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([v_feat, a_feat], dim=-1); mu, logvar = self.mu(x), self.logvar(x)
        std = (0.5 * logvar).exp(); z = mu + torch.randn_like(std) * std
        recon = self.decoder(z); recon_loss = torch.nn.functional.mse_loss(recon, x.detach())
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return {"z": z, "mu": mu, "logvar": logvar, "kl": kl, "recon": recon_loss}
