from __future__ import annotations

import torch


class LocalPolyBasis1D:
    """
    Simple local polynomial basis in normalized coordinate xi=(x-center)/radius:
      [1, xi, xi^2, ..., xi^p]
    """

    def __init__(self, degree: int):
        assert degree >= 0
        self.degree = degree
        self.dim = degree + 1

    def eval(self, x: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """
        x:      [..., 1] or [...]
        center: broadcastable to x
        radius: broadcastable to x (scalar or tensor)
        returns: [..., dim]
        """
        if x.ndim >= 1 and x.shape[-1] == 1:
            x_ = x[..., 0]
        else:
            x_ = x
        xi = (x_ - center) / (radius + 1e-12)

        powers = [torch.ones_like(xi)]
        for k in range(1, self.degree + 1):
            powers.append(powers[-1] * xi)
        return torch.stack(powers, dim=-1)
