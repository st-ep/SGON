from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry.patch_cover_1d import PatchCover1D
from .patch_encoder import PatchEncoder1D
from .gluing_cg import SheafGluingCG, SheafGluingPoly


class SGON1D(nn.Module):
    def __init__(
        self,
        cover: PatchCover1D,
        sensor_x: torch.Tensor,  # [Ns]
        k_sensors_per_patch: int = 16,
        enc_hidden: int = 64,
        use_global: bool = True,
        use_edge_weights: bool = False,
        edge_hidden: int = 32,
        use_deriv_glue: bool = False,
        deriv_weight: float = 1.0,
        use_attention_pool: bool = False,
        use_global_residual: bool = False,
        glue_mode: str = "cg",
        poly_k: int = 3,
        poly_basis: str = "monomial",
        poly_norm: str = "none",
        poly_power_iters: int = 10,
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
            use_attention_pool=use_attention_pool,
            use_global_residual=use_global_residual,
        )

        if glue_mode == "poly":
            self.gluing = SheafGluingPoly(
                src=cover.src,
                dst=cover.dst,
                R_src=cover.R_src,
                R_dst=cover.R_dst,
                R_src_d1=cover.R_src_d1,
                R_dst_d1=cover.R_dst_d1,
                edge_feat=cover.edge_feat,
                use_edge_weights=use_edge_weights,
                edge_hidden=edge_hidden,
                use_deriv=use_deriv_glue,
                deriv_weight=deriv_weight,
                lam=gluing_lam,
                poly_k=poly_k,
                poly_basis=poly_basis,
                poly_norm=poly_norm,
                poly_power_iters=poly_power_iters,
            )
        else:
            self.gluing = SheafGluingCG(
                src=cover.src,
                dst=cover.dst,
                R_src=cover.R_src,
                R_dst=cover.R_dst,
                R_src_d1=cover.R_src_d1,
                R_dst_d1=cover.R_dst_d1,
                edge_feat=cover.edge_feat,
                use_edge_weights=use_edge_weights,
                edge_hidden=edge_hidden,
                use_deriv=use_deriv_glue,
                deriv_weight=deriv_weight,
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
