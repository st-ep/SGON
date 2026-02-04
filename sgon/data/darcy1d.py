from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk


class Darcy1DPreload:
    """
    Preloads Darcy dataset to device and serves fast random batches.

    Expected HF fields per sample:
      - "X": grid points (len N)
      - "u": source term on grid (len N)
      - "s": solution on grid (len N)
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        device: torch.device,
        sensor_size: int,
        n_query: int | None = None,
        seed: int = 0,
    ):
        ds = load_from_disk(data_path)
        assert split in ds, f"split={split} not in dataset splits: {list(ds.keys())}"
        split_ds = ds[split]

        # Grid (assume fixed across samples)
        x = np.asarray(split_ds[0]["X"], dtype=np.float32)  # [N]
        self.x = torch.from_numpy(x).to(device)  # [N]
        self.N = self.x.numel()

        # All u and s
        u_np = np.asarray(split_ds["u"], dtype=np.float32)  # [S,N]
        s_np = np.asarray(split_ds["s"], dtype=np.float32)  # [S,N]
        self.u = torch.from_numpy(u_np).to(device)  # [S,N]
        self.s = torch.from_numpy(s_np).to(device)  # [S,N]
        self.S = self.u.shape[0]

        # Sensors: evenly spaced indices
        sensor_idx = torch.linspace(
            0,
            self.N - 1,
            sensor_size,
            dtype=torch.long,
            device=device,
        )
        self.sensor_idx = sensor_idx  # [Ns]
        self.sensor_x = self.x[self.sensor_idx].view(1, -1, 1)  # [1,Ns,1]

        # Queries: default all points
        if n_query is None:
            query_idx = torch.arange(self.N, device=device)
        else:
            query_idx = torch.linspace(
                0,
                self.N - 1,
                n_query,
                dtype=torch.long,
                device=device,
            )
        self.query_idx = query_idx  # [Nq]
        self.query_x = self.x[self.query_idx].view(1, -1, 1)  # [1,Nq,1]

        # RNG
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def sample_batch(self, batch_size: int):
        """
        Returns:
          xs: [B,Ns,1] sensor locations
          us: [B,Ns,1] u at sensors
          ys: [B,Nq,1] query locations
          s_gt: [B,Nq,1] ground-truth s at queries
          u_q: [B,Nq,1] u at queries (for physics losses)
        """
        device = self.x.device
        idx = torch.randint(0, self.S, (batch_size,), generator=self.gen, device=device)

        u_full = self.u[idx]  # [B,N]
        s_full = self.s[idx]  # [B,N]

        u_sens = u_full[:, self.sensor_idx]  # [B,Ns]
        s_q = s_full[:, self.query_idx]  # [B,Nq]
        u_q = u_full[:, self.query_idx]  # [B,Nq]

        xs = self.sensor_x.expand(batch_size, -1, -1).contiguous()  # [B,Ns,1]
        us = u_sens.unsqueeze(-1)  # [B,Ns,1]
        ys = self.query_x.expand(batch_size, -1, -1).contiguous()  # [B,Nq,1]
        s_gt = s_q.unsqueeze(-1)  # [B,Nq,1]
        u_q = u_q.unsqueeze(-1)  # [B,Nq,1]
        return xs, us, ys, s_gt, u_q
