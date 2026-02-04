from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [B,M,d] -> [B,1,1]
    return (a * b).sum(dim=(1, 2), keepdim=True)


def cg_solve(
    matvec,
    b: torch.Tensor,
    x0: torch.Tensor,
    n_iters: int = 20,
    tol: float = 1e-6,
):
    """
    Conjugate Gradient for SPD systems, batched.
    matvec: function(p)->Ap with p shape [B,M,d]
    """
    x = x0
    r = b - matvec(x)
    p = r
    rsold = batch_dot(r, r)

    for _ in range(n_iters):
        Ap = matvec(p)
        denom = batch_dot(p, Ap) + 1e-12
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = batch_dot(r, r)
        if torch.sqrt(rsnew.mean()).item() < tol:
            break
        p = r + (rsnew / (rsold + 1e-12)) * p
        rsold = rsnew

    return x


class SheafGluingCG(nn.Module):
    """
    Differentiable gluing solve:
      (I + lam * L) c = c0
    where L is induced by edge restrictions (R_src, R_dst).
    """

    def __init__(
        self,
        src: torch.Tensor,  # [E]
        dst: torch.Tensor,  # [E]
        R_src: torch.Tensor,  # [E,A,d]
        R_dst: torch.Tensor,  # [E,A,d]
        R_src_d1: torch.Tensor | None = None,  # [E,A,d]
        R_dst_d1: torch.Tensor | None = None,  # [E,A,d]
        edge_feat: torch.Tensor | None = None,  # [E,F]
        use_edge_weights: bool = False,
        edge_hidden: int = 32,
        use_deriv: bool = False,
        deriv_weight: float = 1.0,
        lam: float = 1.0,
        n_iters: int = 20,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("R_src", R_src)
        self.register_buffer("R_dst", R_dst)
        if R_src_d1 is not None:
            self.register_buffer("R_src_d1", R_src_d1)
        else:
            self.R_src_d1 = None
        if R_dst_d1 is not None:
            self.register_buffer("R_dst_d1", R_dst_d1)
        else:
            self.R_dst_d1 = None
        self.use_edge_weights = use_edge_weights
        self.use_deriv = use_deriv
        self.deriv_weight = deriv_weight
        if edge_feat is not None:
            self.register_buffer("edge_feat", edge_feat)
        else:
            self.edge_feat = None
        if self.use_edge_weights:
            if self.edge_feat is None:
                raise ValueError("edge_feat must be provided when use_edge_weights=True")
            feat_dim = self.edge_feat.shape[-1]
            self.edge_mlp = nn.Sequential(
                nn.Linear(feat_dim, edge_hidden),
                nn.SiLU(),
                nn.Linear(edge_hidden, 1),
            )
        else:
            self.edge_mlp = None
        self.lam = lam
        self.n_iters = n_iters
        self.tol = tol

    def matvec(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: [B,M,d]
        returns: [B,M,d] = p + lam * L p
        """
        B, M, d = p.shape
        E = self.src.numel()
        if E == 0 or self.lam == 0.0:
            return p

        p_src = p[:, self.src, :]  # [B,E,d]
        p_dst = p[:, self.dst, :]  # [B,E,d]

        # r = Rsrc p_src - Rdst p_dst  -> [B,E,A]
        tmp_src = torch.einsum("ead,bed->bea", self.R_src, p_src)
        tmp_dst = torch.einsum("ead,bed->bea", self.R_dst, p_dst)
        r = tmp_src - tmp_dst
        w = None
        if self.use_edge_weights and self.edge_mlp is not None and self.edge_feat.numel() > 0:
            w = F.softplus(self.edge_mlp(self.edge_feat)) + 1e-6  # [E,1]
            r = r * w.view(1, -1, 1).to(r.dtype)

        # contrib to src: Rsrc^T r -> [B,E,d]
        c_src = torch.einsum("ead,bea->bed", self.R_src, r)
        # contrib to dst: -Rdst^T r
        c_dst = -torch.einsum("ead,bea->bed", self.R_dst, r)

        acc = torch.zeros_like(p)
        acc.index_add_(1, self.src, c_src.to(acc.dtype))
        acc.index_add_(1, self.dst, c_dst.to(acc.dtype))

        if self.use_deriv:
            if self.R_src_d1 is None or self.R_dst_d1 is None:
                raise ValueError("R_src_d1/R_dst_d1 must be provided when use_deriv=True")
            tmp_src_d1 = torch.einsum("ead,bed->bea", self.R_src_d1, p_src)
            tmp_dst_d1 = torch.einsum("ead,bed->bea", self.R_dst_d1, p_dst)
            r_d1 = tmp_src_d1 - tmp_dst_d1
            if w is not None:
                r_d1 = r_d1 * w.view(1, -1, 1).to(r_d1.dtype)

            c_src_d1 = torch.einsum("ead,bea->bed", self.R_src_d1, r_d1)
            c_dst_d1 = -torch.einsum("ead,bea->bed", self.R_dst_d1, r_d1)
            acc.index_add_(1, self.src, (self.deriv_weight * c_src_d1).to(acc.dtype))
            acc.index_add_(1, self.dst, (self.deriv_weight * c_dst_d1).to(acc.dtype))

        return p + self.lam * acc

    def forward(self, c0: torch.Tensor) -> torch.Tensor:
        # Solve (I + lam L) c = c0 in float32 for numerical stability.
        c0_f = c0.float()
        x0 = c0_f
        # Disable autocast inside CG to avoid float16 instability.
        if torch.is_autocast_enabled():
            device_type = c0.device.type
            with torch.amp.autocast(device_type=device_type, enabled=False):
                c = cg_solve(self.matvec, b=c0_f, x0=x0, n_iters=self.n_iters, tol=self.tol)
        else:
            c = cg_solve(self.matvec, b=c0_f, x0=x0, n_iters=self.n_iters, tol=self.tol)
        return c.to(c0.dtype)
