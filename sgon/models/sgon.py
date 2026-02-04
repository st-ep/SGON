from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry.patch_cover_1d import PatchCover1D
from .patch_encoder import PatchEncoder1D
from .gluing_cg import SheafGluingCG


class SGON1D(nn.Module):
    def __init__(
        self,
        cover: PatchCover1D,
        sensor_x: torch.Tensor,  # [Ns]
        k_sensors_per_patch: int = 16,
        enc_hidden: int = 64,
        use_global: bool = True,
        gluing_lam: float = 5.0,
        cg_iters: int = 20,
        cg_tol: float = 1e-6,
    ):
        super().__init__()
        # store cover tensors as buffers (so they move with .to(device))
        self.register_buffer("centers", cover.centers)
        self.register_buffer("radius", cover.radius)
        self.register_buffer("w", cover.w)
        self.register_buffer("phi_q", cover.phi_q)

        self.encoder = PatchEncoder1D(
            patch_centers=cover.centers,
            patch_radius=cover.radius,
            sensor_x=sensor_x,
            basis_dim=cover.phi_q.shape[-1],
            k_sensors_per_patch=k_sensors_per_patch,
            hidden=enc_hidden,
            use_global=use_global,
        )

        self.gluing = SheafGluingCG(
            src=cover.src,
            dst=cover.dst,
            R_src=cover.R_src,
            R_dst=cover.R_dst,
            lam=gluing_lam,
            n_iters=cg_iters,
            tol=cg_tol,
        )

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        """
        c: [B,M,d]
        returns s_pred: [B,Q,1]
        """
        s_patch = torch.einsum("mqd,bmd->bmq", self.phi_q, c)
        s = (s_patch * self.w.unsqueeze(0)).sum(dim=1)
        return s.unsqueeze(-1)

    def forward(self, xs: torch.Tensor, us: torch.Tensor):
        """
        xs/us are sensor sets; ys are implicit (cover is built on fixed query points).
        """
        c0 = self.encoder(xs, us)  # [B,M,d]
        c = self.gluing(c0)  # [B,M,d]
        s_pred = self.decode(c)  # [B,Q,1]
        return s_pred, c0, c
