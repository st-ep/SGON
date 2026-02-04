from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [B,M,d] -> [B,1,1]
    return (a * b).sum(dim=(1, 2), keepdim=True)


def laplace_acc(
    p: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    R_src: torch.Tensor,
    R_dst: torch.Tensor,
    R_src_d1: torch.Tensor | None,
    R_dst_d1: torch.Tensor | None,
    edge_feat: torch.Tensor | None,
    edge_mlp: nn.Module | None,
    edge_weight: torch.Tensor | None,
    use_edge_weights: bool,
    use_deriv: bool,
    deriv_weight: float,
) -> torch.Tensor:
    # p: [B,M,d] -> acc: [B,M,d] (Laplacian-like accumulation)
    B, M, d = p.shape
    E = src.numel()
    if E == 0:
        return torch.zeros_like(p)

    p_src = p[:, src, :]  # [B,E,d]
    p_dst = p[:, dst, :]  # [B,E,d]

    tmp_src = torch.einsum("ead,bed->bea", R_src, p_src)
    tmp_dst = torch.einsum("ead,bed->bea", R_dst, p_dst)
    r = tmp_src - tmp_dst

    w = None
    if edge_weight is not None:
        w = edge_weight
        if w.dim() == 2:
            w = w.unsqueeze(-1)
        r = r * w.to(r.dtype)
    elif use_edge_weights and edge_mlp is not None and edge_feat is not None and edge_feat.numel() > 0:
        w = F.softplus(edge_mlp(edge_feat)) + 1e-6  # [E,1]
        r = r * w.view(1, -1, 1).to(r.dtype)

    c_src = torch.einsum("ead,bea->bed", R_src, r)
    c_dst = -torch.einsum("ead,bea->bed", R_dst, r)

    acc = torch.zeros_like(p)
    acc.index_add_(1, src, c_src.to(acc.dtype))
    acc.index_add_(1, dst, c_dst.to(acc.dtype))

    if use_deriv:
        if R_src_d1 is None or R_dst_d1 is None:
            raise ValueError("R_src_d1/R_dst_d1 must be provided when use_deriv=True")
        tmp_src_d1 = torch.einsum("ead,bed->bea", R_src_d1, p_src)
        tmp_dst_d1 = torch.einsum("ead,bed->bea", R_dst_d1, p_dst)
        r_d1 = tmp_src_d1 - tmp_dst_d1
        if w is not None:
            r_d1 = r_d1 * w.view(1, -1, 1).to(r_d1.dtype)

        c_src_d1 = torch.einsum("ead,bea->bed", R_src_d1, r_d1)
        c_dst_d1 = -torch.einsum("ead,bea->bed", R_dst_d1, r_d1)
        acc.index_add_(1, src, (deriv_weight * c_src_d1).to(acc.dtype))
        acc.index_add_(1, dst, (deriv_weight * c_dst_d1).to(acc.dtype))

    return acc


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
        self._edge_weight: torch.Tensor | None = None

    def matvec(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: [B,M,d]
        returns: [B,M,d] = p + lam * L p
        """
        if self.src.numel() == 0 or self.lam == 0.0:
            return p

        acc = laplace_acc(
            p=p,
            src=self.src,
            dst=self.dst,
            R_src=self.R_src,
            R_dst=self.R_dst,
            R_src_d1=self.R_src_d1,
            R_dst_d1=self.R_dst_d1,
            edge_feat=self.edge_feat,
            edge_mlp=self.edge_mlp,
            edge_weight=self._edge_weight,
            use_edge_weights=self.use_edge_weights,
            use_deriv=self.use_deriv,
            deriv_weight=self.deriv_weight,
        )
        return p + self.lam * acc

    def forward(self, c0: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        # Solve (I + lam L) c = c0 in float32 for numerical stability.
        c0_f = c0.float()
        x0 = c0_f
        self._edge_weight = edge_weight
        # Disable autocast inside CG to avoid float16 instability.
        if torch.is_autocast_enabled():
            device_type = c0.device.type
            with torch.amp.autocast(device_type=device_type, enabled=False):
                c = cg_solve(self.matvec, b=c0_f, x0=x0, n_iters=self.n_iters, tol=self.tol)
        else:
            c = cg_solve(self.matvec, b=c0_f, x0=x0, n_iters=self.n_iters, tol=self.tol)
        self._edge_weight = None
        return c.to(c0.dtype)


class SheafGluingPoly(nn.Module):
    """
    Learned polynomial filter in L (sheaf Laplacian) as a fast alternative to CG.
    c = sum_{k=0}^K a_k (lam L)^k c0
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
        poly_k: int = 3,
        poly_basis: str = "monomial",
        poly_norm: str = "none",
        poly_power_iters: int = 10,
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
        self.poly_k = poly_k
        self.poly_basis = poly_basis
        self.poly_norm = poly_norm
        self.poly_power_iters = poly_power_iters
        coeffs = torch.zeros(poly_k + 1)
        coeffs[0] = 1.0
        self.poly_coeffs = nn.Parameter(coeffs)
        self._lmax_cached: torch.Tensor | None = None
        self._edge_weight: torch.Tensor | None = None

    def _apply_L(self, x: torch.Tensor) -> torch.Tensor:
        return laplace_acc(
            p=x,
            src=self.src,
            dst=self.dst,
            R_src=self.R_src,
            R_dst=self.R_dst,
            R_src_d1=self.R_src_d1,
            R_dst_d1=self.R_dst_d1,
            edge_feat=self.edge_feat,
            edge_mlp=self.edge_mlp,
            edge_weight=self._edge_weight,
            use_edge_weights=self.use_edge_weights,
            use_deriv=self.use_deriv,
            deriv_weight=self.deriv_weight,
        )

    def _estimate_lmax(self, c0: torch.Tensor) -> torch.Tensor:
        # Power iteration for spectral norm of L (not including lam).
        M = c0.shape[1]
        d = c0.shape[2]
        device_type = c0.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x = torch.randn(1, M, d, device=c0.device, dtype=torch.float32)
            x = x / (x.norm(dim=(1, 2), keepdim=True) + 1e-12)
            n = torch.tensor(1.0, device=c0.device, dtype=torch.float32)
            iters = max(1, int(self.poly_power_iters))
            for _ in range(iters):
                x = self._apply_L(x)
                n = x.norm(dim=(1, 2), keepdim=True)
                x = x / (n + 1e-12)
            return n.clamp_min(1e-6)

    def _get_lmax(self, c0: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._estimate_lmax(c0)
        if self._lmax_cached is None or self._lmax_cached.device != c0.device:
            self._lmax_cached = self._estimate_lmax(c0).detach()
        return self._lmax_cached

    def forward(self, c0: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        self._edge_weight = edge_weight
        if self.src.numel() == 0 or self.lam == 0.0 or self.poly_k == 0:
            self._edge_weight = None
            return c0

        # Compute polynomial in (lam * L)
        def apply_L(x: torch.Tensor) -> torch.Tensor:
            return self._apply_L(x)

        if self.poly_basis == "chebyshev":
            scale = 1.0
            if self.poly_norm == "degree":
                deg = torch.bincount(self.src, minlength=int(c0.shape[1])).float()
                if deg.numel() > 0:
                    max_deg = deg.max().clamp_min(1.0)
                    scale = 1.0 / max_deg
            elif self.poly_norm == "power":
                scale = 1.0 / self._get_lmax(c0)

            # Chebyshev recurrence: T0 = c0, T1 = Lhat c0, T_{k+1} = 2 Lhat T_k - T_{k-1}
            Lhat = lambda x: (self.lam * scale) * apply_L(x)
            T0 = c0
            out = self.poly_coeffs[0] * T0
            if self.poly_k >= 1:
                T1 = Lhat(c0)
                out = out + self.poly_coeffs[1] * T1
                for k in range(2, self.poly_k + 1):
                    T2 = 2.0 * Lhat(T1) - T0
                    out = out + self.poly_coeffs[k] * T2
                    T0, T1 = T1, T2
            self._edge_weight = None
            return out

        # Monomial basis
        scale = 1.0
        if self.poly_norm == "degree":
            deg = torch.bincount(self.src, minlength=int(c0.shape[1])).float()
            if deg.numel() > 0:
                max_deg = deg.max().clamp_min(1.0)
                scale = 1.0 / max_deg
        elif self.poly_norm == "power":
            scale = 1.0 / self._get_lmax(c0)

        out = self.poly_coeffs[0] * c0
        v = c0
        for k in range(1, self.poly_k + 1):
            v = (self.lam * scale) * apply_L(v)
            out = out + self.poly_coeffs[k] * v
        self._edge_weight = None
        return out
