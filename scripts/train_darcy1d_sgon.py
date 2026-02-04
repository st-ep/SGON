from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import trange

from sgon.utils.seed import set_seed
from sgon.utils.metrics import rel_l2
from sgon.data.darcy1d import Darcy1DPreload
from sgon.geometry.patch_cover_1d import PatchCover1D
from sgon.models.sgon import SGON1D
from sgon.models.sgon_multiscale import SGON1DCoarseFine, SGON1DThreeScale


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        type=str,
        default="Data/darcy_1d_data/darcy_1d_dataset_501",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string, e.g. cpu, cuda, cuda:0, cuda:1",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="Enable AMP (can cause NaNs)")

    # data
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sensor_size", type=int, default=64)
    p.add_argument("--n_query", type=int, default=None)

    # SGON cover
    p.add_argument("--n_patches", type=int, default=64)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--basis_type", type=str, default="poly", choices=["poly", "rbf"])
    p.add_argument("--rbf_centers", type=int, default=8)
    p.add_argument("--rbf_gamma", type=float, default=2.0)
    p.add_argument("--radius_factor", type=float, default=1.5)
    p.add_argument("--anchors_per_edge", type=int, default=8)
    p.add_argument("--multiscale", action="store_true", help="Enable coarse-to-fine model")
    p.add_argument("--n_patches_coarse", type=int, default=16)
    p.add_argument("--degree_coarse", type=int, default=None)
    p.add_argument("--n_patches_mid", type=int, default=0)
    p.add_argument("--degree_mid", type=int, default=None)
    p.add_argument("--basis_type_coarse", type=str, default=None, choices=["poly", "rbf"])
    p.add_argument("--basis_type_mid", type=str, default=None, choices=["poly", "rbf"])

    # model + glue
    p.add_argument("--k_sensors_per_patch", type=int, default=64)
    p.add_argument("--enc_hidden", type=int, default=32)
    p.add_argument("--no_global", action="store_true", help="Disable global sensor pooling")
    p.add_argument("--edge_weights", action="store_true", help="Use learned edge weights")
    p.add_argument("--edge_hidden", type=int, default=32)
    p.add_argument("--deriv_glue", action="store_true", help="Use value+derivative gluing")
    p.add_argument("--deriv_weight", type=float, default=1.0)
    p.add_argument("--attention_pool", action="store_true", help="Use attention-weighted sensor pooling")
    p.add_argument("--global_residual", action="store_true", help="Add global latent residual to patch coeffs")
    p.add_argument("--u_backbone", action="store_true", help="Use small 1D conv backbone over sensor u")
    p.add_argument("--u_backbone_pos", action="store_true", help="Add sensor x as a channel to u_backbone Conv1d")
    p.add_argument("--u_backbone_channels", type=int, default=16)
    p.add_argument("--u_backbone_layers", type=int, default=4)
    p.add_argument("--u_backbone_kernel", type=int, default=5)
    p.add_argument("--glue_mode", type=str, default="cg", choices=["cg", "poly"])
    p.add_argument("--poly_k", type=int, default=3)
    p.add_argument("--poly_basis", type=str, default="monomial", choices=["monomial", "chebyshev"])
    p.add_argument("--poly_norm", type=str, default="none", choices=["none", "degree", "power"])
    p.add_argument("--poly_power_iters", type=int, default=10)
    p.add_argument("--gluing_lam", type=float, default=0.0)
    p.add_argument("--cg_iters", type=int, default=20)

    # optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--physics_beta", type=float, default=0.0)
    p.add_argument("--train_noise_std", type=float, default=0.0, help="Std of Gaussian noise added to sensor u during training")
    p.add_argument(
        "--lr_schedule_steps",
        type=int,
        nargs="+",
        default=[50000, 100000, 150000, 175000, 1250000, 1500000],
        help="List of steps for LR decay milestones.",
    )
    p.add_argument(
        "--lr_schedule_gammas",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5],
        help="List of multiplicative factors for LR decay.",
    )
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--run_name", type=str, default="darcy1d")
    p.add_argument("--save_every", type=int, default=0, help="Save periodic checkpoints (0 disables).")

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    set_seed(args.seed)

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = Path(args.log_dir) / args.run_name / timestamp
    log_root.mkdir(parents=True, exist_ok=True)
    with open(log_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

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
        basis_type=args.basis_type,
        rbf_centers=args.rbf_centers,
        rbf_gamma=args.rbf_gamma,
        radius_factor=args.radius_factor,
        anchors_per_edge=args.anchors_per_edge,
        device=device,
    )
    if args.multiscale:
        basis_type_coarse = args.basis_type if args.basis_type_coarse is None else args.basis_type_coarse
        degree_coarse = args.degree if args.degree_coarse is None else args.degree_coarse
        cover_coarse = PatchCover1D.build(
            y_query=y_query,
            n_patches=args.n_patches_coarse,
            degree=degree_coarse,
            basis_type=basis_type_coarse,
            rbf_centers=args.rbf_centers,
            rbf_gamma=args.rbf_gamma,
            radius_factor=args.radius_factor,
            anchors_per_edge=args.anchors_per_edge,
            device=device,
        )
        if args.n_patches_mid > 0:
            basis_type_mid = args.basis_type if args.basis_type_mid is None else args.basis_type_mid
            degree_mid = args.degree if args.degree_mid is None else args.degree_mid
            cover_mid = PatchCover1D.build(
                y_query=y_query,
                n_patches=args.n_patches_mid,
                degree=degree_mid,
                basis_type=basis_type_mid,
                rbf_centers=args.rbf_centers,
                rbf_gamma=args.rbf_gamma,
                radius_factor=args.radius_factor,
                anchors_per_edge=args.anchors_per_edge,
                device=device,
            )

    # Model
    sensor_x = train.sensor_x[0, :, 0]  # [Ns]
    if args.multiscale:
        # Map sensor locations to nearest query indices for coarse conditioning
        yq = y_query[:, 0] if y_query.ndim >= 1 and y_query.shape[-1] == 1 else y_query
        sx = sensor_x
        dist = (sx[:, None] - yq[None, :]).abs()
        sensor_to_query = dist.argmin(dim=1)
        if args.n_patches_mid > 0:
            model = SGON1DThreeScale(
                cover_coarse=cover_coarse,
                cover_mid=cover_mid,
                cover_fine=cover,
                sensor_x=sensor_x,
                sensor_to_query=sensor_to_query,
                k_sensors_per_patch=args.k_sensors_per_patch,
                enc_hidden=args.enc_hidden,
                use_global=not args.no_global,
                use_edge_weights=args.edge_weights,
                edge_hidden=args.edge_hidden,
                use_deriv_glue=args.deriv_glue,
                deriv_weight=args.deriv_weight,
                use_attention_pool=args.attention_pool,
                use_global_residual=args.global_residual,
                use_u_backbone=args.u_backbone,
                use_u_backbone_pos=args.u_backbone_pos,
                u_backbone_channels=args.u_backbone_channels,
                u_backbone_layers=args.u_backbone_layers,
                u_backbone_kernel=args.u_backbone_kernel,
                glue_mode=args.glue_mode,
                poly_k=args.poly_k,
                poly_basis=args.poly_basis,
                poly_norm=args.poly_norm,
                poly_power_iters=args.poly_power_iters,
                gluing_lam=args.gluing_lam,
                cg_iters=args.cg_iters,
            ).to(device)
        else:
            model = SGON1DCoarseFine(
                cover_coarse=cover_coarse,
                cover_fine=cover,
                sensor_x=sensor_x,
                sensor_to_query=sensor_to_query,
                k_sensors_per_patch=args.k_sensors_per_patch,
                enc_hidden=args.enc_hidden,
                use_global=not args.no_global,
                use_edge_weights=args.edge_weights,
                edge_hidden=args.edge_hidden,
                use_deriv_glue=args.deriv_glue,
                deriv_weight=args.deriv_weight,
                use_attention_pool=args.attention_pool,
                use_global_residual=args.global_residual,
                use_u_backbone=args.u_backbone,
                use_u_backbone_pos=args.u_backbone_pos,
                u_backbone_channels=args.u_backbone_channels,
                u_backbone_layers=args.u_backbone_layers,
                u_backbone_kernel=args.u_backbone_kernel,
                glue_mode=args.glue_mode,
                poly_k=args.poly_k,
                poly_basis=args.poly_basis,
                poly_norm=args.poly_norm,
                poly_power_iters=args.poly_power_iters,
                gluing_lam=args.gluing_lam,
                cg_iters=args.cg_iters,
            ).to(device)
    else:
        model = SGON1D(
            cover=cover,
            sensor_x=sensor_x,
            k_sensors_per_patch=args.k_sensors_per_patch,
            enc_hidden=args.enc_hidden,
            use_global=not args.no_global,
            use_edge_weights=args.edge_weights,
            edge_hidden=args.edge_hidden,
            use_deriv_glue=args.deriv_glue,
            deriv_weight=args.deriv_weight,
            use_attention_pool=args.attention_pool,
            use_global_residual=args.global_residual,
            use_u_backbone=args.u_backbone,
            use_u_backbone_pos=args.u_backbone_pos,
            u_backbone_channels=args.u_backbone_channels,
            u_backbone_layers=args.u_backbone_layers,
            u_backbone_kernel=args.u_backbone_kernel,
            glue_mode=args.glue_mode,
            poly_k=args.poly_k,
            poly_basis=args.poly_basis,
            poly_norm=args.poly_norm,
            poly_power_iters=args.poly_power_iters,
            gluing_lam=args.gluing_lam,
            cg_iters=args.cg_iters,
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError(
            "lr_schedule_steps and lr_schedule_gammas must have the same length."
        )

    def lr_lambda(step: int) -> float:
        scale = 1.0
        for s, g in zip(args.lr_schedule_steps, args.lr_schedule_gammas):
            if step >= s:
                scale *= g
        return scale

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    pbar = trange(1, args.steps + 1, desc="train", leave=True)
    x_query = train.query_x[0].detach()  # [Q,1]
    best_rel = float("inf")
    history = []

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
        if args.train_noise_std > 0.0:
            noise = torch.randn_like(us)
            us = us + args.train_noise_std * noise

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
        scheduler.step()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                # Train metrics
                train_rel = rel_l2(s_pred, s_gt).item()

                # Test metrics
                xs_t, us_t, ys_t, s_gt_t, _ = test.sample_batch(min(256, args.batch_size))
                s_pred_t, _, _ = model(xs_t, us_t)
                err = rel_l2(s_pred_t, s_gt_t).item()
                test_mse = torch.mean((s_pred_t - s_gt_t) ** 2).item()
            model.train()
            postfix = {
                "train_mse": f"{mse.item():.4e}",
                "train_rel_l2": f"{train_rel:.4e}",
                "test_mse": f"{test_mse:.4e}",
                "test_rel_l2": f"{err:.4e}",
            }
            if phys is not None:
                postfix["phys"] = f"{phys.item():.4e}"
            pbar.set_postfix(postfix)

            record = {
                "step": step,
                "train_mse": float(mse.item()),
                "train_rel_l2": float(train_rel),
                "test_mse": float(test_mse),
                "test_rel_l2": float(err),
            }
            if phys is not None:
                record["phys"] = float(phys.item())
            history.append(record)
            with open(log_root / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if err < best_rel:
                best_rel = err
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "step": step,
                        "best_rel_l2": best_rel,
                        "args": vars(args),
                    },
                    log_root / "best.pt",
                )

        if args.save_every > 0 and step % args.save_every == 0:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "step": step,
                    "args": vars(args),
                },
                log_root / f"ckpt_step_{step}.pt",
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "step": args.steps,
            "args": vars(args),
        },
        log_root / "last.pt",
    )

    print(f"Done. Logs saved to: {log_root}")


if __name__ == "__main__":
    main()
