from __future__ import annotations

from dataclasses import dataclass
import torch

from .basis_1d import LocalPolyBasis1D, LocalRBFBasis1D


@dataclass
class PatchCover1D:
    # Geometry
    centers: torch.Tensor  # [M]
    radius: torch.Tensor  # scalar tensor
    # Partition of unity weights for decoding
    w: torch.Tensor  # [M,Q]
    # Basis evaluated at query points
    phi_q: torch.Tensor  # [M,Q,d]
    # Overlap graph
    src: torch.Tensor  # [E]
    dst: torch.Tensor  # [E]
    # Restriction matrices (basis evaluated at anchors inside overlaps)
    R_src: torch.Tensor  # [E,A,d]
    R_dst: torch.Tensor  # [E,A,d]
    R_src_d1: torch.Tensor  # [E,A,d]
    R_dst_d1: torch.Tensor  # [E,A,d]
    # Edge features for optional learned weights
    edge_feat: torch.Tensor  # [E,F]

    @staticmethod
    def build(
        y_query: torch.Tensor,  # [Q,1] or [Q]
        n_patches: int,
        degree: int,
        basis_type: str = "poly",
        rbf_centers: int = 8,
        rbf_gamma: float = 2.0,
        radius_factor: float = 1.5,
        sigma_factor: float = 0.5,
        anchors_per_edge: int = 8,
        device: torch.device | None = None,
    ) -> "PatchCover1D":
        """
        For sign-of-life on 1D [0,1]: choose evenly spaced centers and fixed radius.
        Connect adjacent patches with overlap.
        Anchors are selected from the shared query points in the overlap region.
        """
        if device is None:
            device = y_query.device
        y = y_query
        if y.ndim >= 1 and y.shape[-1] == 1:
            y = y[:, 0]
        y = y.to(device)
        Q = y.numel()

        y_min = y.min()
        y_max = y.max()
        centers = torch.linspace(y_min, y_max, n_patches, device=device)  # [M]
        if n_patches > 1:
            spacing = centers[1] - centers[0]
        else:
            spacing = (y_max - y_min)

        radius = radius_factor * spacing
        sigma = sigma_factor * radius

        # Distances: [M,Q]
        dist = (y[None, :] - centers[:, None]).abs()
        mask = dist <= radius

        # Partition of unity weights (masked RBFs, normalized across patches)
        w = torch.exp(-0.5 * (dist / (sigma + 1e-12)) ** 2) * mask
        w = w / (w.sum(dim=0, keepdim=True) + 1e-12)

        # Basis eval on query points
        if basis_type == "rbf":
            basis = LocalRBFBasis1D(n_centers=rbf_centers, gamma=rbf_gamma)
        else:
            basis = LocalPolyBasis1D(degree=degree)
        phi_q = basis.eval(
            x=y[None, :, None].expand(n_patches, Q, 1),
            center=centers[:, None].expand(n_patches, Q),
            radius=radius,
        )

        # Overlap edges: connect adjacent patches if they overlap
        src_list = []
        dst_list = []
        R_src_list = []
        R_dst_list = []
        R_src_d1_list = []
        R_dst_d1_list = []
        edge_feat_list = []

        for j in range(n_patches - 1):
            inter = torch.nonzero(mask[j] & mask[j + 1], as_tuple=False).squeeze(-1)
            if inter.numel() == 0:
                continue
            # Choose anchors_per_edge anchors evenly across the intersection indices
            if inter.numel() <= anchors_per_edge:
                anchor_idx = inter
            else:
                t = torch.linspace(0, inter.numel() - 1, anchors_per_edge, device=device)
                anchor_idx = inter[t.long()]

            # Restriction matrices = basis at anchor points for each patch
            y_a = y[anchor_idx]
            Rj = basis.eval(x=y_a[:, None], center=centers[j], radius=radius)
            Rk = basis.eval(x=y_a[:, None], center=centers[j + 1], radius=radius)
            Rj_d1 = basis.eval_d1(x=y_a[:, None], center=centers[j], radius=radius)
            Rk_d1 = basis.eval_d1(x=y_a[:, None], center=centers[j + 1], radius=radius)

            # Edge features: normalized center distance, overlap fraction, anchor fraction
            dist_c = (centers[j + 1] - centers[j]).abs() / (radius + 1e-12)
            if inter.numel() >= 2:
                overlap_len = (y[inter[-1]] - y[inter[0]]).abs()
            else:
                overlap_len = torch.tensor(0.0, device=device)
            overlap_frac = overlap_len / (2.0 * radius + 1e-12)
            anchor_frac = torch.tensor(
                float(anchor_idx.numel()) / float(anchors_per_edge), device=device
            )
            edge_feat_list.append(torch.stack([dist_c, overlap_frac, anchor_frac]))

            src_list.append(j)
            dst_list.append(j + 1)
            R_src_list.append(Rj)
            R_dst_list.append(Rk)
            R_src_d1_list.append(Rj_d1)
            R_dst_d1_list.append(Rk_d1)

        if len(src_list) == 0:
            src = torch.zeros(0, dtype=torch.long, device=device)
            dst = torch.zeros(0, dtype=torch.long, device=device)
            R_src = torch.zeros(0, anchors_per_edge, basis.dim, device=device)
            R_dst = torch.zeros(0, anchors_per_edge, basis.dim, device=device)
            R_src_d1 = torch.zeros(0, anchors_per_edge, basis.dim, device=device)
            R_dst_d1 = torch.zeros(0, anchors_per_edge, basis.dim, device=device)
            edge_feat = torch.zeros(0, 3, device=device)
        else:
            src = torch.tensor(src_list, dtype=torch.long, device=device)
            dst = torch.tensor(dst_list, dtype=torch.long, device=device)
            A = anchors_per_edge
            d = basis.dim
            E = len(src_list)
            R_src = torch.zeros(E, A, d, device=device)
            R_dst = torch.zeros(E, A, d, device=device)
            R_src_d1 = torch.zeros(E, A, d, device=device)
            R_dst_d1 = torch.zeros(E, A, d, device=device)
            for e in range(E):
                a_e = R_src_list[e].shape[0]
                R_src[e, :a_e] = R_src_list[e]
                R_dst[e, :a_e] = R_dst_list[e]
                R_src_d1[e, :a_e] = R_src_d1_list[e]
                R_dst_d1[e, :a_e] = R_dst_d1_list[e]
            edge_feat = torch.stack(edge_feat_list, dim=0)

        return PatchCover1D(
            centers=centers,
            radius=radius,
            w=w,
            phi_q=phi_q,
            src=src,
            dst=dst,
            R_src=R_src,
            R_dst=R_dst,
            R_src_d1=R_src_d1,
            R_dst_d1=R_dst_d1,
            edge_feat=edge_feat,
        )
