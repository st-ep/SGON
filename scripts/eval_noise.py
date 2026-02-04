from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from sgon.data.darcy1d import Darcy1DPreload
from sgon.geometry.patch_cover_1d import PatchCover1D
from sgon.models.sgon import SGON1D
from sgon.models.sgon_multiscale import SGON1DCoarseFine, SGON1DThreeScale


def rel_l2_per_sample(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.norm(pred - true, dim=(1, 2))
    den = torch.norm(true, dim=(1, 2)) + eps
    return num / den


def mse_per_sample(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - true) ** 2, dim=(1, 2))


def build_model_from_ckpt(ckpt: dict, data: Darcy1DPreload, device: torch.device):
    cfg = ckpt.get("args", {})

    def get(name, default):
        return cfg[name] if name in cfg else default

    # Build cover(s)
    y_query = data.query_x[0]
    cover = PatchCover1D.build(
        y_query=y_query,
        n_patches=get("n_patches", 64),
        degree=get("degree", 3),
        basis_type=get("basis_type", "poly"),
        rbf_centers=get("rbf_centers", 8),
        rbf_gamma=get("rbf_gamma", 2.0),
        radius_factor=get("radius_factor", 1.5),
        anchors_per_edge=get("anchors_per_edge", 8),
        device=device,
    )

    multiscale = get("multiscale", False)
    n_patches_mid = get("n_patches_mid", 0)
    cover_coarse = None
    cover_mid = None
    if multiscale:
        degree_coarse = get("degree_coarse", None)
        if degree_coarse is None:
            degree_coarse = get("degree", 3)
        basis_type_coarse = get("basis_type_coarse", None)
        if basis_type_coarse is None:
            basis_type_coarse = get("basis_type", "poly")
        cover_coarse = PatchCover1D.build(
            y_query=y_query,
            n_patches=get("n_patches_coarse", 16),
            degree=degree_coarse,
            basis_type=basis_type_coarse,
            rbf_centers=get("rbf_centers", 8),
            rbf_gamma=get("rbf_gamma", 2.0),
            radius_factor=get("radius_factor", 1.5),
            anchors_per_edge=get("anchors_per_edge", 8),
            device=device,
        )
        if n_patches_mid > 0:
            degree_mid = get("degree_mid", None)
            if degree_mid is None:
                degree_mid = get("degree", 3)
            basis_type_mid = get("basis_type_mid", None)
            if basis_type_mid is None:
                basis_type_mid = get("basis_type", "poly")
            cover_mid = PatchCover1D.build(
                y_query=y_query,
                n_patches=n_patches_mid,
                degree=degree_mid,
                basis_type=basis_type_mid,
                rbf_centers=get("rbf_centers", 8),
                rbf_gamma=get("rbf_gamma", 2.0),
                radius_factor=get("radius_factor", 1.5),
                anchors_per_edge=get("anchors_per_edge", 8),
                device=device,
            )

    # Model
    sensor_x = data.sensor_x[0, :, 0]
    use_global = not get("no_global", False)
    if multiscale:
        yq = y_query[:, 0] if y_query.ndim >= 1 and y_query.shape[-1] == 1 else y_query
        dist = (sensor_x[:, None] - yq[None, :]).abs()
        sensor_to_query = dist.argmin(dim=1)
        if n_patches_mid > 0 and cover_mid is not None:
            model = SGON1DThreeScale(
                cover_coarse=cover_coarse,
                cover_mid=cover_mid,
                cover_fine=cover,
                sensor_x=sensor_x,
                sensor_to_query=sensor_to_query,
                k_sensors_per_patch=get("k_sensors_per_patch", 32),
                enc_hidden=get("enc_hidden", 64),
                use_global=use_global,
                use_edge_weights=get("edge_weights", False),
                edge_hidden=get("edge_hidden", 32),
                use_deriv_glue=get("deriv_glue", False),
                deriv_weight=get("deriv_weight", 1.0),
                gluing_lam=get("gluing_lam", 1.0),
                cg_iters=get("cg_iters", 20),
            ).to(device)
        else:
            model = SGON1DCoarseFine(
                cover_coarse=cover_coarse,
                cover_fine=cover,
                sensor_x=sensor_x,
                sensor_to_query=sensor_to_query,
                k_sensors_per_patch=get("k_sensors_per_patch", 32),
                enc_hidden=get("enc_hidden", 64),
                use_global=use_global,
                use_edge_weights=get("edge_weights", False),
                edge_hidden=get("edge_hidden", 32),
                use_deriv_glue=get("deriv_glue", False),
                deriv_weight=get("deriv_weight", 1.0),
                gluing_lam=get("gluing_lam", 1.0),
                cg_iters=get("cg_iters", 20),
            ).to(device)
    else:
        model = SGON1D(
            cover=cover,
            sensor_x=sensor_x,
            k_sensors_per_patch=get("k_sensors_per_patch", 32),
            enc_hidden=get("enc_hidden", 64),
            use_global=use_global,
            use_edge_weights=get("edge_weights", False),
            edge_hidden=get("edge_hidden", 32),
            use_deriv_glue=get("deriv_glue", False),
            deriv_weight=get("deriv_weight", 1.0),
            gluing_lam=get("gluing_lam", 1.0),
            cg_iters=get("cg_iters", 20),
        ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, default="Data/darcy_1d_data/darcy_1d_dataset_501")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1])
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--benchmark_name", type=str, default="darcy1d_noise")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    ckpt = torch.load(args.ckpt, map_location=device)
    train_cfg = ckpt.get("args", {})
    sensor_size = train_cfg.get("sensor_size", 64)
    n_query = train_cfg.get("n_query", None)

    # Data (match training sensor/query config)
    data = Darcy1DPreload(
        data_path=args.data_path,
        split="test",
        device=device,
        sensor_size=sensor_size,
        n_query=n_query,
        seed=args.seed,
    )

    # Build model from checkpoint
    model, train_cfg = build_model_from_ckpt(ckpt, data, device)

    # Fixed eval indices
    n_total = data.S
    n_eval = min(args.n_eval, n_total)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    idx = torch.randperm(n_total, generator=gen, device=device)[:n_eval]

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_root = Path(args.log_dir) / args.benchmark_name / timestamp
    log_root.mkdir(parents=True, exist_ok=True)
    with open(log_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval": vars(args),
                "train": train_cfg,
                "ckpt": str(Path(args.ckpt).resolve()),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    results = []
    for li, nl in enumerate(args.noise_levels):
        noise_gen = torch.Generator(device=device)
        noise_gen.manual_seed(args.seed + 1000 + li)

        rel_vals = []
        mse_vals = []
        for start in range(0, n_eval, args.batch_size):
            batch_idx = idx[start : start + args.batch_size]
            u_full = data.u[batch_idx]
            s_full = data.s[batch_idx]
            u_sens = u_full[:, data.sensor_idx]
            s_q = s_full[:, data.query_idx]
            xs = data.sensor_x.expand(batch_idx.numel(), -1, -1).contiguous()
            us = u_sens.unsqueeze(-1)
            s_gt = s_q.unsqueeze(-1)

            if nl > 0.0:
                noise = torch.randn(us.shape, device=us.device, dtype=us.dtype, generator=noise_gen)
                us = us + nl * noise

            with torch.no_grad():
                s_pred, _, _ = model(xs, us)
            rel = rel_l2_per_sample(s_pred, s_gt)
            mse = mse_per_sample(s_pred, s_gt)
            rel_vals.append(rel)
            mse_vals.append(mse)

        rel_all = torch.cat(rel_vals)
        mse_all = torch.cat(mse_vals)
        results.append(
            {
                "noise": float(nl),
                "rel_l2_mean": float(rel_all.mean().item()),
                "rel_l2_std": float(rel_all.std().item()),
                "mse_mean": float(mse_all.mean().item()),
                "mse_std": float(mse_all.std().item()),
            }
        )

    with open(log_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved noise eval results to: {log_root}")


if __name__ == "__main__":
    main()
