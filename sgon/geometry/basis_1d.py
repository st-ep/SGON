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

    def eval_d1(self, x: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """
        First derivative d/dx of the local polynomial basis.
        returns: [..., dim]
        """
        if x.ndim >= 1 and x.shape[-1] == 1:
            x_ = x[..., 0]
        else:
            x_ = x
        xi = (x_ - center) / (radius + 1e-12)
        inv_r = 1.0 / (radius + 1e-12)

        deriv = [torch.zeros_like(xi)]
        if self.degree >= 1:
            prev = torch.ones_like(xi)  # xi^0
            for k in range(1, self.degree + 1):
                deriv.append(k * prev * inv_r)
                prev = prev * xi
        return torch.stack(deriv, dim=-1)


class LocalRBFBasis1D:
    """
    Gaussian RBF basis on normalized coordinate xi=(x-center)/radius.
    Centers are fixed and evenly spaced in [-1, 1].
    """

    def __init__(self, n_centers: int, gamma: float = 2.0):
        assert n_centers >= 1
        self.n_centers = n_centers
        self.gamma = gamma
        self.dim = n_centers
        self.centers = torch.linspace(-1.0, 1.0, n_centers)

    def _phi(self, xi: torch.Tensor) -> torch.Tensor:
        # xi: [...]
        # returns [..., n_centers]
        c = self.centers.to(xi.device, xi.dtype)
        diff = xi[..., None] - c
        return torch.exp(-self.gamma * diff * diff)

    def eval(self, x: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        if x.ndim >= 1 and x.shape[-1] == 1:
            x_ = x[..., 0]
        else:
            x_ = x
        xi = (x_ - center) / (radius + 1e-12)
        return self._phi(xi)

    def eval_d1(self, x: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        if x.ndim >= 1 and x.shape[-1] == 1:
            x_ = x[..., 0]
        else:
            x_ = x
        xi = (x_ - center) / (radius + 1e-12)
        inv_r = 1.0 / (radius + 1e-12)
        c = self.centers.to(xi.device, xi.dtype)
        diff = xi[..., None] - c
        phi = torch.exp(-self.gamma * diff * diff)
        return (-2.0 * self.gamma * diff * phi) * inv_r
