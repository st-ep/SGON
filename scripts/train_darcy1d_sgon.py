from __future__ import annotations

import argparse
import torch
from tqdm import trange

from sgon.utils.seed import set_seed
from sgon.utils.metrics import rel_l2
from sgon.data.darcy1d import Darcy1DPreload
from sgon.geometry.patch_cover_1d import PatchCover1D
from sgon.models.sgon import SGON1D


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        type=str,
        default="Data/darcy_1d_data/darcy_1d_dataset_501",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="Enable AMP (can cause NaNs)")

    # data
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sensor_size", type=int, default=64)
    p.add_argument("--n_query", type=int, default=None)

    # SGON cover
    p.add_argument("--n_patches", type=int, default=64)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--radius_factor", type=float, default=1.5)
    p.add_argument("--anchors_per_edge", type=int, default=8)

    # model + glue
    p.add_argument("--k_sensors_per_patch", type=int, default=16)
    p.add_argument("--enc_hidden", type=int, default=64)
    p.add_argument("--no_global", action="store_true", help="Disable global sensor pooling")
    p.add_argument("--gluing_lam", type=float, default=5.0)
    p.add_argument("--cg_iters", type=int, default=20)

    # optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--physics_beta", type=float, default=0.0)

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Data
    train = Darcy1DPreload(
        data_path=args.data_path,
        split="train",
        device=device,
        sensor_size=args.sensor_size,
        n_query=args.n_query,
        seed=args.seed,
    )
    test = Darcy1DPreload(
        data_path=args.data_path,
        split="test",
        device=device,
        sensor_size=args.sensor_size,
        n_query=args.n_query,
        seed=args.seed + 123,
    )

    # Build patch cover ONCE on the query grid
    y_query = train.query_x[0]  # [Q,1]
    cover = PatchCover1D.build(
        y_query=y_query,
        n_patches=args.n_patches,
        degree=args.degree,
        radius_factor=args.radius_factor,
        anchors_per_edge=args.anchors_per_edge,
        device=device,
    )

    # Model
    sensor_x = train.sensor_x[0, :, 0]  # [Ns]
    model = SGON1D(
        cover=cover,
        sensor_x=sensor_x,
        k_sensors_per_patch=args.k_sensors_per_patch,
        enc_hidden=args.enc_hidden,
        use_global=not args.no_global,
        gluing_lam=args.gluing_lam,
        cg_iters=args.cg_iters,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    pbar = trange(1, args.steps + 1, desc="train", leave=True)
    x_query = train.query_x[0].detach()  # [Q,1]

    def darcy_residual_loss(s_pred: torch.Tensor, u_q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # s_pred/u_q: [B,Q,1], x: [Q,1] or [Q]
        s = s_pred[..., 0]
        u = u_q[..., 0]
        x_ = x[:, 0] if x.ndim >= 1 and x.shape[-1] == 1 else x
        dx = x_[1] - x_[0]
        s_left = s[:, :-2]
        s_mid = s[:, 1:-1]
        s_right = s[:, 2:]
        k_left = 0.2 + (0.5 * (s_left + s_mid)) ** 2
        k_right = 0.2 + (0.5 * (s_mid + s_right)) ** 2
        flux_left = k_left * (s_mid - s_left) / dx
        flux_right = k_right * (s_right - s_mid) / dx
        r = -(flux_right - flux_left) / dx - u[:, 1:-1]
        r_bc0 = s[:, 0]
        r_bc1 = s[:, -1]
        return r.pow(2).mean() + 0.5 * (r_bc0.pow(2).mean() + r_bc1.pow(2).mean())

    for step in pbar:
        xs, us, ys, s_gt, u_q = train.sample_batch(args.batch_size)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            s_pred, c0, c = model(xs, us)
            mse = torch.mean((s_pred - s_gt) ** 2)
            if args.physics_beta > 0.0:
                phys = darcy_residual_loss(s_pred, u_q, x_query)
                loss = mse + args.physics_beta * phys
            else:
                phys = None
                loss = mse

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                xs_t, us_t, ys_t, s_gt_t, _ = test.sample_batch(min(256, args.batch_size))
                s_pred_t, _, _ = model(xs_t, us_t)
                err = rel_l2(s_pred_t, s_gt_t).item()
            model.train()
            postfix = {"train_mse": f"{mse.item():.4e}", "test_rel_l2": f"{err:.4e}"}
            if phys is not None:
                postfix["phys"] = f"{phys.item():.4e}"
            pbar.set_postfix(postfix)

    print("Done.")


if __name__ == "__main__":
    main()
