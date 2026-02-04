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
from sgon.models.deeponet import DeepONetWrapper


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
    p.add_argument("--amp", action="store_true", help="Enable AMP")

    # data
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sensor_size", type=int, default=64)
    p.add_argument("--n_query", type=int, default=None)

    # model
    p.add_argument("--p", type=int, default=32)
    p.add_argument("--branch_hidden", type=int, default=256)
    p.add_argument("--trunk_hidden", type=int, default=256)
    p.add_argument("--n_branch_layers", type=int, default=4)
    p.add_argument("--n_trunk_layers", type=int, default=4)

    # optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--eval_every", type=int, default=250)
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

    # logging
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--run_name", type=str, default="darcy1d_deeponet")
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

    # Model
    model = DeepONetWrapper(
        branch_input_dim=args.sensor_size,
        trunk_input_dim=1,
        p=args.p,
        trunk_hidden_size=args.trunk_hidden,
        n_trunk_layers=args.n_trunk_layers,
        branch_hidden_size=args.branch_hidden,
        n_branch_layers=args.n_branch_layers,
        initial_lr=args.lr,
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
    best_rel = float("inf")
    history = []

    for step in pbar:
        xs, us, ys, s_gt, _ = train.sample_batch(args.batch_size)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            s_pred = model(xs, us, ys)
            mse = torch.mean((s_pred - s_gt) ** 2)
            loss = mse

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                train_rel = rel_l2(s_pred, s_gt).item()

                xs_t, us_t, ys_t, s_gt_t, _ = test.sample_batch(min(256, args.batch_size))
                s_pred_t = model(xs_t, us_t, ys_t)
                err = rel_l2(s_pred_t, s_gt_t).item()
                test_mse = torch.mean((s_pred_t - s_gt_t) ** 2).item()
            model.train()
            postfix = {
                "train_mse": f"{mse.item():.4e}",
                "train_rel_l2": f"{train_rel:.4e}",
                "test_mse": f"{test_mse:.4e}",
                "test_rel_l2": f"{err:.4e}",
            }
            pbar.set_postfix(postfix)

            record = {
                "step": step,
                "train_mse": float(mse.item()),
                "train_rel_l2": float(train_rel),
                "test_mse": float(test_mse),
                "test_rel_l2": float(err),
            }
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
