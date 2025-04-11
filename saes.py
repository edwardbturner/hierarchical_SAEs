import torch
import torch.nn.functional as F
from torch import nn


class VanillaSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_coefficient=0.01):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_coefficient = sparsity_coefficient

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

    def loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        reconstruction_loss = F.mse_loss(x_recon, x)
        sparsity_loss = self.sparsity_coefficient * torch.mean(torch.abs(z))
        return reconstruction_loss + sparsity_loss


class MarginalSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_0=1e-3, lambda_abs=1e-2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.lambda_0 = lambda_0
        self.lambda_abs = lambda_abs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def l1_absorption_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """
        For each row r_j in z, compute a weighted sum of L1 distances to other rows,
        with weights determined by softmin to ensure differentiability.
        """
        B = z.size(0)
        diffs = z.unsqueeze(1) - z.unsqueeze(0)  # (B, B, k)
        pairwise_l1 = torch.sum(torch.abs(diffs), dim=-1)  # (B, B)

        # Create mask for self-comparisons
        mask = torch.eye(B, dtype=torch.bool, device=z.device)
        pairwise_l1 = pairwise_l1.masked_fill(mask, float("inf"))

        # Use softmin instead of min for differentiability
        weights = F.softmin(pairwise_l1, dim=1)  # (B, B)
        weighted_dists = torch.sum(weights * pairwise_l1, dim=1)  # (B,)

        return torch.sum(weighted_dists)

    def loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(x_recon, x)
        sparsity_loss = self.lambda_0 * torch.mean(torch.abs(z))  # torch.count_nonzero(z).float() if want l0 sparsity
        absorption = self.lambda_abs * self.l1_absorption_penalty(z)
        return recon_loss + sparsity_loss + absorption
