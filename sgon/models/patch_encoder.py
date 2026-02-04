from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class PatchEncoder1D(nn.Module):
    """
    Maps sensor set (x_i, u_i) to initial patch coefficients c0_j.

    For scalability: precompute nearest sensor indices per patch.
    """

    def __init__(
        self,
        patch_centers: torch.Tensor,  # [M]
        patch_radius: torch.Tensor,  # scalar
        sensor_x: torch.Tensor,  # [Ns]
        basis_dim: int,
        k_sensors_per_patch: int = 16,
        hidden: int = 64,
        use_global: bool = True,
    ):
        super().__init__()
        self.register_buffer("centers", patch_centers.detach().clone())
        self.register_buffer("radius", patch_radius.detach().clone())
        self.M = self.centers.numel()
        self.use_global = use_global

        # Precompute K nearest sensors per patch (on init)
        dist = (sensor_x[None, :] - self.centers[:, None]).abs()
        knn = torch.topk(
            dist,
            k=min(k_sensors_per_patch, sensor_x.numel()),
            largest=False,
        ).indices
        self.register_buffer("sensor_ids", knn)
        rel = (sensor_x[knn] - self.centers[:, None]) / (self.radius + 1e-12)
        self.register_buffer("rel_x", rel.unsqueeze(-1))

        # Pointwise sensor MLP: input [dx, u] -> hidden
        self.mlp_point = mlp([2, hidden, hidden])
        if self.use_global:
            # Global sensor MLP: input [x, u] -> hidden
            s = sensor_x
            s_norm = 2.0 * (s - s.min()) / (s.max() - s.min() + 1e-12) - 1.0
            self.register_buffer("sensor_x_norm", s_norm.view(-1, 1))
            self.mlp_global = mlp([2, hidden, hidden])

        # Patch MLP: [pooled_hidden, center] -> coeffs
        in_dim = hidden + 1 + (hidden if self.use_global else 0)
        self.mlp_patch = mlp([in_dim, hidden, basis_dim])

    def forward(self, xs: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
        """
        xs: [B,Ns,1]  (not used except for sanity; we assume fixed sensors)
        us: [B,Ns,1]
        returns c0: [B,M,d]
        """
        B, Ns, _ = us.shape
        u = us[..., 0]  # [B,Ns]

        # Gather u at K sensors per patch -> [B,M,K]
        u_patch = u[:, self.sensor_ids]

        # Build per-sensor features: concat(rel_x, u)
        rel_x = self.rel_x.unsqueeze(0).expand(B, -1, -1, -1)
        u_feat = u_patch.unsqueeze(-1)
        feat = torch.cat([rel_x, u_feat], dim=-1)

        h = self.mlp_point(feat)  # [B,M,K,H]
        pooled = h.mean(dim=2)  # [B,M,H]

        # Add center as feature (normalized to roughly [-1,1])
        c = self.centers
        c_norm = 2.0 * (c - c.min()) / (c.max() - c.min() + 1e-12) - 1.0
        c_norm = c_norm.view(1, -1, 1).expand(B, -1, -1)

        if self.use_global:
            s_norm = self.sensor_x_norm.unsqueeze(0).expand(B, -1, -1)  # [B,Ns,1]
            g_in = torch.cat([s_norm, us], dim=-1)  # [B,Ns,2]
            g = self.mlp_global(g_in).mean(dim=1)  # [B,H]
            g = g.unsqueeze(1).expand(B, self.M, -1)  # [B,M,H]
            z = torch.cat([pooled, c_norm, g], dim=-1)
        else:
            z = torch.cat([pooled, c_norm], dim=-1)
        c0 = self.mlp_patch(z)
        return c0
