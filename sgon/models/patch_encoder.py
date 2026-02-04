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
        use_u_backbone: bool = False,
        use_u_backbone_pos: bool = False,
        u_backbone_channels: int = 16,
        u_backbone_layers: int = 4,
        u_backbone_kernel: int = 5,
    ):
        super().__init__()
        self.register_buffer("centers", patch_centers.detach().clone())
        self.register_buffer("radius", patch_radius.detach().clone())
        self.M = self.centers.numel()
        self.use_global = use_global
        self.extra_sensor_dim = extra_sensor_dim
        self.use_attention_pool = use_attention_pool
        self.use_global_residual = use_global_residual
        self.use_u_backbone = use_u_backbone
        self.use_u_backbone_pos = use_u_backbone_pos

        if self.use_global or self.use_u_backbone_pos:
            s = sensor_x
            s_norm = 2.0 * (s - s.min()) / (s.max() - s.min() + 1e-12) - 1.0
            self.register_buffer("sensor_x_norm", s_norm.view(-1, 1))

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

        # Optional 1D conv backbone over ordered sensors to provide global context.
        # This assumes sensor ordering is consistent across batches (fixed sensors).
        conv_feat_dim = 0
        if self.use_u_backbone:
            if u_backbone_kernel % 2 == 0:
                raise ValueError("u_backbone_kernel must be odd for 'same' padding.")
            cin = 1 + self.extra_sensor_dim + (1 if self.use_u_backbone_pos else 0)
            ch = int(u_backbone_channels)
            layers = []
            dilation = 1
            for i in range(int(u_backbone_layers)):
                in_ch = cin if i == 0 else ch
                pad = dilation * (u_backbone_kernel - 1) // 2
                layers.append(
                    nn.Conv1d(
                        in_ch,
                        ch,
                        kernel_size=u_backbone_kernel,
                        padding=pad,
                        dilation=dilation,
                    )
                )
                layers.append(nn.SiLU())
                dilation *= 2
            self.u_backbone = nn.Sequential(*layers)
            conv_feat_dim = ch
        else:
            self.u_backbone = None

        # Pointwise sensor MLP: input [dx, u, u_backbone..., extra...] -> hidden
        sensor_feat_dim = 2 + conv_feat_dim + self.extra_sensor_dim
        self.mlp_point = mlp([sensor_feat_dim, hidden, hidden])
        if self.use_attention_pool:
            self.mlp_attn = mlp([sensor_feat_dim, hidden, 1])
        if self.use_global:
            # Global sensor MLP: input [x, u] -> hidden
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
        if self.u_backbone is not None:
            x_ch = None
            if self.use_u_backbone_pos:
                x_ch = self.sensor_x_norm.unsqueeze(0).expand(B, -1, -1)  # [B,Ns,1]
            if self.extra_sensor_dim > 0:
                conv_in = torch.cat([us, extra_sensor_feat], dim=-1)  # [B,Ns,1+F]
            else:
                conv_in = us  # [B,Ns,1]
            if x_ch is not None:
                conv_in = torch.cat([x_ch, conv_in], dim=-1)  # [B,Ns,1+Cin]
            conv_in = conv_in.permute(0, 2, 1).contiguous()  # [B,Cin,Ns]
            conv_out = self.u_backbone(conv_in)  # [B,C,Ns]
            conv_out = conv_out.permute(0, 2, 1).contiguous()  # [B,Ns,C]
            conv_patch = conv_out[:, self.sensor_ids, :]  # [B,M,K,C]

        # Build per-sensor features: concat(rel_x, u)
        rel_x = self.rel_x.unsqueeze(0).expand(B, -1, -1, -1)
        u_feat = u_patch.unsqueeze(-1)
        if self.extra_sensor_dim > 0:
            if self.u_backbone is not None:
                feat = torch.cat([rel_x, u_feat, conv_patch, extra_patch], dim=-1)
            else:
                feat = torch.cat([rel_x, u_feat, extra_patch], dim=-1)
        else:
            if self.u_backbone is not None:
                feat = torch.cat([rel_x, u_feat, conv_patch], dim=-1)
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
