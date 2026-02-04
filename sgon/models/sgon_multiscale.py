from __future__ import annotations

import torch
import torch.nn as nn

from ..geometry.patch_cover_1d import PatchCover1D
from .patch_encoder import PatchEncoder1D
from .gluing_cg import SheafGluingCG, SheafGluingPoly


class SGON1DCoarseFine(nn.Module):
    """
    Two-scale SGON: coarse prediction conditions the fine encoder.

    Coarse -> fine conditioning uses coarse field values at sensor locations
    as an extra per-sensor feature for the fine patch encoder.
    """

    def __init__(
        self,
        cover_coarse: PatchCover1D,
        cover_fine: PatchCover1D,
        sensor_x: torch.Tensor,  # [Ns]
        sensor_to_query: torch.Tensor,  # [Ns]
        k_sensors_per_patch: int = 16,
        enc_hidden: int = 64,
        use_global: bool = True,
        use_edge_weights: bool = False,
        edge_hidden: int = 32,
        use_deriv_glue: bool = False,
        deriv_weight: float = 1.0,
        use_attention_pool: bool = False,
        use_global_residual: bool = False,
        use_u_backbone: bool = False,
        use_u_backbone_pos: bool = False,
        u_backbone_channels: int = 16,
        u_backbone_layers: int = 4,
        u_backbone_kernel: int = 5,
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
        self.register_buffer("sensor_to_query", sensor_to_query)

        # Coarse cover buffers
        self.register_buffer("c_centers", cover_coarse.centers)
        self.register_buffer("c_radius", cover_coarse.radius)
        self.register_buffer("c_w", cover_coarse.w)
        self.register_buffer("c_phi_q", cover_coarse.phi_q)

        # Fine cover buffers
        self.register_buffer("f_centers", cover_fine.centers)
        self.register_buffer("f_radius", cover_fine.radius)
        self.register_buffer("f_w", cover_fine.w)
        self.register_buffer("f_phi_q", cover_fine.phi_q)

        # Coarse encoder + gluing
        self.coarse_encoder = PatchEncoder1D(
            patch_centers=cover_coarse.centers,
            patch_radius=cover_coarse.radius,
            sensor_x=sensor_x,
            basis_dim=cover_coarse.phi_q.shape[-1],
            k_sensors_per_patch=k_sensors_per_patch,
            hidden=enc_hidden,
            use_global=use_global,
            extra_sensor_dim=0,
            use_attention_pool=use_attention_pool,
            use_global_residual=use_global_residual,
            use_u_backbone=use_u_backbone,
            use_u_backbone_pos=use_u_backbone_pos,
            u_backbone_channels=u_backbone_channels,
            u_backbone_layers=u_backbone_layers,
            u_backbone_kernel=u_backbone_kernel,
        )
        if glue_mode == "poly":
            self.coarse_gluing = SheafGluingPoly(
                src=cover_coarse.src,
                dst=cover_coarse.dst,
                R_src=cover_coarse.R_src,
                R_dst=cover_coarse.R_dst,
                R_src_d1=cover_coarse.R_src_d1,
                R_dst_d1=cover_coarse.R_dst_d1,
                edge_feat=cover_coarse.edge_feat,
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
            self.coarse_gluing = SheafGluingCG(
                src=cover_coarse.src,
                dst=cover_coarse.dst,
                R_src=cover_coarse.R_src,
                R_dst=cover_coarse.R_dst,
                R_src_d1=cover_coarse.R_src_d1,
                R_dst_d1=cover_coarse.R_dst_d1,
                edge_feat=cover_coarse.edge_feat,
                use_edge_weights=use_edge_weights,
                edge_hidden=edge_hidden,
                use_deriv=use_deriv_glue,
                deriv_weight=deriv_weight,
                lam=gluing_lam,
                n_iters=cg_iters,
                tol=cg_tol,
            )

        # Fine encoder + gluing
        self.fine_encoder = PatchEncoder1D(
            patch_centers=cover_fine.centers,
            patch_radius=cover_fine.radius,
            sensor_x=sensor_x,
            basis_dim=cover_fine.phi_q.shape[-1],
            k_sensors_per_patch=k_sensors_per_patch,
            hidden=enc_hidden,
            use_global=use_global,
            extra_sensor_dim=1,
            use_attention_pool=use_attention_pool,
            use_global_residual=use_global_residual,
            use_u_backbone=use_u_backbone,
            use_u_backbone_pos=use_u_backbone_pos,
            u_backbone_channels=u_backbone_channels,
            u_backbone_layers=u_backbone_layers,
            u_backbone_kernel=u_backbone_kernel,
        )
        if glue_mode == "poly":
            self.fine_gluing = SheafGluingPoly(
                src=cover_fine.src,
                dst=cover_fine.dst,
                R_src=cover_fine.R_src,
                R_dst=cover_fine.R_dst,
                R_src_d1=cover_fine.R_src_d1,
                R_dst_d1=cover_fine.R_dst_d1,
                edge_feat=cover_fine.edge_feat,
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
            self.fine_gluing = SheafGluingCG(
                src=cover_fine.src,
                dst=cover_fine.dst,
                R_src=cover_fine.R_src,
                R_dst=cover_fine.R_dst,
                R_src_d1=cover_fine.R_src_d1,
                R_dst_d1=cover_fine.R_dst_d1,
                edge_feat=cover_fine.edge_feat,
                use_edge_weights=use_edge_weights,
                edge_hidden=edge_hidden,
                use_deriv=use_deriv_glue,
                deriv_weight=deriv_weight,
                lam=gluing_lam,
                n_iters=cg_iters,
                tol=cg_tol,
            )

    @staticmethod
    def decode(c: torch.Tensor, w: torch.Tensor, phi_q: torch.Tensor) -> torch.Tensor:
        s_patch = torch.einsum("mqd,bmd->bmq", phi_q, c)
        s = (s_patch * w.unsqueeze(0)).sum(dim=1)
        return s.unsqueeze(-1)

    def forward(self, xs: torch.Tensor, us: torch.Tensor):
        # Coarse pass
        c0_c = self.coarse_encoder(xs, us)
        c_c = self.coarse_gluing(c0_c)
        s_coarse = self.decode(c_c, self.c_w, self.c_phi_q)  # [B,Q,1]

        # Coarse->fine conditioning at sensor locations
        s_coarse_s = s_coarse[:, self.sensor_to_query, :]  # [B,Ns,1]

        # Fine pass
        c0_f = self.fine_encoder(xs, us, extra_sensor_feat=s_coarse_s)
        c_f = self.fine_gluing(c0_f)
        s_fine = self.decode(c_f, self.f_w, self.f_phi_q)
        return s_fine, c0_f, c_f


class SGON1DThreeScale(nn.Module):
    """
    Three-scale SGON: coarse -> mid -> fine conditioning.

    Coarse prediction conditions the mid encoder; coarse+mid condition the fine encoder.
    """

    def __init__(
        self,
        cover_coarse: PatchCover1D,
        cover_mid: PatchCover1D,
        cover_fine: PatchCover1D,
        sensor_x: torch.Tensor,  # [Ns]
        sensor_to_query: torch.Tensor,  # [Ns]
        k_sensors_per_patch: int = 16,
        enc_hidden: int = 64,
        use_global: bool = True,
        use_edge_weights: bool = False,
        edge_hidden: int = 32,
        use_deriv_glue: bool = False,
        deriv_weight: float = 1.0,
        use_attention_pool: bool = False,
        use_global_residual: bool = False,
        use_u_backbone: bool = False,
        use_u_backbone_pos: bool = False,
        u_backbone_channels: int = 16,
        u_backbone_layers: int = 4,
        u_backbone_kernel: int = 5,
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
        self.register_buffer("sensor_to_query", sensor_to_query)

        # Coarse cover buffers
        self.register_buffer("c_centers", cover_coarse.centers)
        self.register_buffer("c_radius", cover_coarse.radius)
        self.register_buffer("c_w", cover_coarse.w)
        self.register_buffer("c_phi_q", cover_coarse.phi_q)

        # Mid cover buffers
        self.register_buffer("m_centers", cover_mid.centers)
        self.register_buffer("m_radius", cover_mid.radius)
        self.register_buffer("m_w", cover_mid.w)
        self.register_buffer("m_phi_q", cover_mid.phi_q)

        # Fine cover buffers
        self.register_buffer("f_centers", cover_fine.centers)
        self.register_buffer("f_radius", cover_fine.radius)
        self.register_buffer("f_w", cover_fine.w)
        self.register_buffer("f_phi_q", cover_fine.phi_q)

        # Coarse encoder + gluing
        self.coarse_encoder = PatchEncoder1D(
            patch_centers=cover_coarse.centers,
            patch_radius=cover_coarse.radius,
            sensor_x=sensor_x,
            basis_dim=cover_coarse.phi_q.shape[-1],
            k_sensors_per_patch=k_sensors_per_patch,
            hidden=enc_hidden,
            use_global=use_global,
            extra_sensor_dim=0,
            use_attention_pool=use_attention_pool,
            use_global_residual=use_global_residual,
            use_u_backbone=use_u_backbone,
            use_u_backbone_pos=use_u_backbone_pos,
            u_backbone_channels=u_backbone_channels,
            u_backbone_layers=u_backbone_layers,
            u_backbone_kernel=u_backbone_kernel,
        )
        if glue_mode == "poly":
            self.coarse_gluing = SheafGluingPoly(
                src=cover_coarse.src,
                dst=cover_coarse.dst,
                R_src=cover_coarse.R_src,
                R_dst=cover_coarse.R_dst,
                R_src_d1=cover_coarse.R_src_d1,
                R_dst_d1=cover_coarse.R_dst_d1,
                edge_feat=cover_coarse.edge_feat,
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
            self.coarse_gluing = SheafGluingCG(
                src=cover_coarse.src,
                dst=cover_coarse.dst,
                R_src=cover_coarse.R_src,
                R_dst=cover_coarse.R_dst,
                R_src_d1=cover_coarse.R_src_d1,
                R_dst_d1=cover_coarse.R_dst_d1,
                edge_feat=cover_coarse.edge_feat,
                use_edge_weights=use_edge_weights,
                edge_hidden=edge_hidden,
                use_deriv=use_deriv_glue,
                deriv_weight=deriv_weight,
                lam=gluing_lam,
                n_iters=cg_iters,
                tol=cg_tol,
            )

        # Mid encoder + gluing (conditioned on coarse)
        self.mid_encoder = PatchEncoder1D(
            patch_centers=cover_mid.centers,
            patch_radius=cover_mid.radius,
            sensor_x=sensor_x,
            basis_dim=cover_mid.phi_q.shape[-1],
            k_sensors_per_patch=k_sensors_per_patch,
            hidden=enc_hidden,
            use_global=use_global,
            extra_sensor_dim=1,
            use_attention_pool=use_attention_pool,
            use_global_residual=use_global_residual,
            use_u_backbone=use_u_backbone,
            use_u_backbone_pos=use_u_backbone_pos,
            u_backbone_channels=u_backbone_channels,
            u_backbone_layers=u_backbone_layers,
            u_backbone_kernel=u_backbone_kernel,
        )
        if glue_mode == "poly":
            self.mid_gluing = SheafGluingPoly(
                src=cover_mid.src,
                dst=cover_mid.dst,
                R_src=cover_mid.R_src,
                R_dst=cover_mid.R_dst,
                R_src_d1=cover_mid.R_src_d1,
                R_dst_d1=cover_mid.R_dst_d1,
                edge_feat=cover_mid.edge_feat,
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
            self.mid_gluing = SheafGluingCG(
                src=cover_mid.src,
                dst=cover_mid.dst,
                R_src=cover_mid.R_src,
                R_dst=cover_mid.R_dst,
                R_src_d1=cover_mid.R_src_d1,
                R_dst_d1=cover_mid.R_dst_d1,
                edge_feat=cover_mid.edge_feat,
                use_edge_weights=use_edge_weights,
                edge_hidden=edge_hidden,
                use_deriv=use_deriv_glue,
                deriv_weight=deriv_weight,
                lam=gluing_lam,
                n_iters=cg_iters,
                tol=cg_tol,
            )

        # Fine encoder + gluing (conditioned on coarse + mid)
        self.fine_encoder = PatchEncoder1D(
            patch_centers=cover_fine.centers,
            patch_radius=cover_fine.radius,
            sensor_x=sensor_x,
            basis_dim=cover_fine.phi_q.shape[-1],
            k_sensors_per_patch=k_sensors_per_patch,
            hidden=enc_hidden,
            use_global=use_global,
            extra_sensor_dim=2,
            use_attention_pool=use_attention_pool,
            use_global_residual=use_global_residual,
            use_u_backbone=use_u_backbone,
            use_u_backbone_pos=use_u_backbone_pos,
            u_backbone_channels=u_backbone_channels,
            u_backbone_layers=u_backbone_layers,
            u_backbone_kernel=u_backbone_kernel,
        )
        if glue_mode == "poly":
            self.fine_gluing = SheafGluingPoly(
                src=cover_fine.src,
                dst=cover_fine.dst,
                R_src=cover_fine.R_src,
                R_dst=cover_fine.R_dst,
                R_src_d1=cover_fine.R_src_d1,
                R_dst_d1=cover_fine.R_dst_d1,
                edge_feat=cover_fine.edge_feat,
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
            self.fine_gluing = SheafGluingCG(
                src=cover_fine.src,
                dst=cover_fine.dst,
                R_src=cover_fine.R_src,
                R_dst=cover_fine.R_dst,
                R_src_d1=cover_fine.R_src_d1,
                R_dst_d1=cover_fine.R_dst_d1,
                edge_feat=cover_fine.edge_feat,
                use_edge_weights=use_edge_weights,
                edge_hidden=edge_hidden,
                use_deriv=use_deriv_glue,
                deriv_weight=deriv_weight,
                lam=gluing_lam,
                n_iters=cg_iters,
                tol=cg_tol,
            )

    @staticmethod
    def decode(c: torch.Tensor, w: torch.Tensor, phi_q: torch.Tensor) -> torch.Tensor:
        s_patch = torch.einsum("mqd,bmd->bmq", phi_q, c)
        s = (s_patch * w.unsqueeze(0)).sum(dim=1)
        return s.unsqueeze(-1)

    def forward(self, xs: torch.Tensor, us: torch.Tensor):
        # Coarse
        c0_c = self.coarse_encoder(xs, us)
        c_c = self.coarse_gluing(c0_c)
        s_c = self.decode(c_c, self.c_w, self.c_phi_q)  # [B,Q,1]
        s_c_s = s_c[:, self.sensor_to_query, :]  # [B,Ns,1]

        # Mid
        c0_m = self.mid_encoder(xs, us, extra_sensor_feat=s_c_s)
        c_m = self.mid_gluing(c0_m)
        s_m = self.decode(c_m, self.m_w, self.m_phi_q)
        s_m_s = s_m[:, self.sensor_to_query, :]  # [B,Ns,1]

        # Fine
        extra_f = torch.cat([s_c_s, s_m_s], dim=-1)  # [B,Ns,2]
        c0_f = self.fine_encoder(xs, us, extra_sensor_feat=extra_f)
        c_f = self.fine_gluing(c0_f)
        s_f = self.decode(c_f, self.f_w, self.f_phi_q)
        return s_f, c0_f, c_f
