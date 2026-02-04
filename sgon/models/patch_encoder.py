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
        extra_sensor_dim: int = 0,
        use_attention_pool: bool = False,
        use_global_residual: bool = False,
    ):
        super().__init__()
        self.register_buffer("centers", patch_centers.detach().clone())
        self.register_buffer("radius", patch_radius.detach().clone())
        self.M = self.centers.numel()
        self.use_global = use_global
        self.extra_sensor_dim = extra_sensor_dim
        self.use_attention_pool = use_attention_pool
        self.use_global_residual = use_global_residual

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

        # Pointwise sensor MLP: input [dx, u, extra...] -> hidden
        sensor_feat_dim = 2 + self.extra_sensor_dim
        self.mlp_point = mlp([sensor_feat_dim, hidden, hidden])
        if self.use_attention_pool:
            self.mlp_attn = mlp([sensor_feat_dim, hidden, 1])
        if self.use_global:
            # Global sensor MLP: input [x, u] -> hidden
            s = sensor_x
            s_norm = 2.0 * (s - s.min()) / (s.max() - s.min() + 1e-12) - 1.0
            self.register_buffer("sensor_x_norm", s_norm.view(-1, 1))
            global_feat_dim = 2 + self.extra_sensor_dim
            self.mlp_global = mlp([global_feat_dim, hidden, hidden])

        # Patch MLP: [pooled_hidden, center] -> coeffs
        in_dim = hidden + 1 + (hidden if self.use_global else 0)
        self.mlp_patch = mlp([in_dim, hidden, basis_dim])
        if self.use_global_residual:
            g_in_dim = hidden + 1  # global pooled + center
            self.mlp_global_proj = mlp([g_in_dim, hidden, basis_dim])

    def forward(
        self,
        xs: torch.Tensor,
        us: torch.Tensor,
        extra_sensor_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        xs: [B,Ns,1]  (not used except for sanity; we assume fixed sensors)
        us: [B,Ns,1]
        returns c0: [B,M,d]
        """
        B, Ns, _ = us.shape
        u = us[..., 0]  # [B,Ns]
        if self.extra_sensor_dim > 0:
            if extra_sensor_feat is None:
                extra_sensor_feat = torch.zeros(
                    B,
                    Ns,
                    self.extra_sensor_dim,
                    device=us.device,
                    dtype=us.dtype,
                )
            elif extra_sensor_feat.shape[-1] != self.extra_sensor_dim:
                raise ValueError("extra_sensor_feat has wrong feature dimension")

        # Gather u at K sensors per patch -> [B,M,K]
        u_patch = u[:, self.sensor_ids]
        if self.extra_sensor_dim > 0:
            extra_patch = extra_sensor_feat[:, self.sensor_ids, :]  # [B,M,K,F]

        # Build per-sensor features: concat(rel_x, u)
        rel_x = self.rel_x.unsqueeze(0).expand(B, -1, -1, -1)
        u_feat = u_patch.unsqueeze(-1)
        if self.extra_sensor_dim > 0:
            feat = torch.cat([rel_x, u_feat, extra_patch], dim=-1)
        else:
            feat = torch.cat([rel_x, u_feat], dim=-1)

        h = self.mlp_point(feat)  # [B,M,K,H]
        if self.use_attention_pool:
            a = self.mlp_attn(feat).squeeze(-1)  # [B,M,K]
            a = torch.softmax(a, dim=-1).unsqueeze(-1)
            pooled = (h * a).sum(dim=2)
        else:
            pooled = h.mean(dim=2)  # [B,M,H]

        # Add center as feature (normalized to roughly [-1,1])
        c = self.centers
        c_norm = 2.0 * (c - c.min()) / (c.max() - c.min() + 1e-12) - 1.0
        c_norm = c_norm.view(1, -1, 1).expand(B, -1, -1)

        g = None
        if self.use_global:
            s_norm = self.sensor_x_norm.unsqueeze(0).expand(B, -1, -1)  # [B,Ns,1]
            if self.extra_sensor_dim > 0:
                g_in = torch.cat([s_norm, us, extra_sensor_feat], dim=-1)
            else:
                g_in = torch.cat([s_norm, us], dim=-1)
            g = self.mlp_global(g_in).mean(dim=1)  # [B,H]
            g = g.unsqueeze(1).expand(B, self.M, -1)  # [B,M,H]
            z = torch.cat([pooled, c_norm, g], dim=-1)
        else:
            z = torch.cat([pooled, c_norm], dim=-1)

        c0 = self.mlp_patch(z)
        if self.use_global_residual and g is not None:
            g_proj = self.mlp_global_proj(torch.cat([g, c_norm], dim=-1))
            c0 = c0 + g_proj
        return c0
