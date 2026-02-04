import torch


@torch.no_grad()
def rel_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # pred/true: [B,Q,1]
    num = torch.norm(pred - true, dim=(1, 2))
    den = torch.norm(true, dim=(1, 2)) + eps
    return (num / den).mean()
