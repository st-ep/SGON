from __future__ import annotations

import torch
import torch.nn as nn


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
        lam: float = 1.0,
        n_iters: int = 20,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("R_src", R_src)
        self.register_buffer("R_dst", R_dst)
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

        # contrib to src: Rsrc^T r -> [B,E,d]
        c_src = torch.einsum("ead,bea->bed", self.R_src, r)
        # contrib to dst: -Rdst^T r
        c_dst = -torch.einsum("ead,bea->bed", self.R_dst, r)

        acc = torch.zeros_like(p)
        acc.index_add_(1, self.src, c_src.to(acc.dtype))
        acc.index_add_(1, self.dst, c_dst.to(acc.dtype))

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
