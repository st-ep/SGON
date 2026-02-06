# --- File: generate_dataset.py ---
"""
HOW TO RUN (minimal example)
----------------------------
# Default run (N=64, L=1, nu=1e-3, T=1, dt=2e-3, CNAB2, Kolmogorov forcing)
python generate_dataset.py --output-root ./data --dataset-name ns2d_snapshot

# Example: make it harder (lower viscosity, longer horizon, higher resolution)
python generate_dataset.py --output-root ./data --dataset-name ns2d_hard \
  --N 128 --nu 1e-4 --T 2.0 --dt 1e-3 --ic-kmax 16 --ic-rms 7.0 --forcing-amp 0.2

Notes:
- Output is saved locally as a Hugging Face DatasetDict with splits: train, test.
- Metadata is saved to <output_root>/<dataset_name>/metadata.json
"""

from __future__ import annotations

import argparse
import os

from ns2d.dataset_writer import DatasetBuildConfig, build_and_save_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 2D Navier–Stokes (vorticity) snapshot-map dataset (ω0 → ωT).")

    # Output / dataset
    p.add_argument("--output-root", type=str, default="./data", help="Root directory to save datasets.")
    p.add_argument("--dataset-name", type=str, default="ns2d_snapshot", help="Dataset folder name under output-root.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory if it exists.")

    p.add_argument("--num-samples", type=int, default=1000, help="Total number of samples (train+test).")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio in (0,1).")
    p.add_argument("--seed", type=int, default=1234, help="Global seed (reproducible split + per-sample RNG).")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Stored dtype.")

    # Grid / PDE parameters
    p.add_argument("--N", type=int, default=64, help="Grid resolution (NxN).")
    p.add_argument("--L", type=float, default=1.0, help="Domain size, periodic on [0,L)x[0,L).")
    p.add_argument("--nu", type=float, default=1e-3, help="Viscosity ν (fixed for all samples).")
    p.add_argument("--T", type=float, default=1.0, help="Final snapshot time T.")

    # Time stepping controls
    p.add_argument("--dt", type=float, default=2e-3, help="Requested time step (treated as max dt).")
    p.add_argument(
        "--integrator",
        type=str,
        default="cnab2",
        choices=["cnab2", "etdrk4"],
        help="Time integrator: cnab2 (fast) or etdrk4 (more accurate).",
    )
    p.add_argument("--dealias", action="store_true", help="Enable 2/3-rule dealiasing (recommended).")
    p.add_argument("--no-dealias", dest="dealias", action="store_false", help="Disable dealiasing.")
    p.set_defaults(dealias=True)

    p.add_argument(
        "--adjust-dt-to-T",
        action="store_true",
        help="Adjust dt to divide T exactly with integer steps (recommended; default on).",
    )
    p.add_argument("--no-adjust-dt-to-T", dest="adjust_dt_to_T", action="store_false")
    p.set_defaults(adjust_dt_to_T=True)

    # Optional dt stability helper (computed once per sample from initial velocity)
    p.add_argument("--dt-auto", action="store_true", help="Compute dt <= dt based on initial CFL (constant dt).")
    p.add_argument("--cfl", type=float, default=0.5, help="CFL number for dt-auto.")

    # Fail-fast checks
    p.add_argument("--check-nan", action="store_true", help="Enable NaN/Inf checks during time stepping.")
    p.add_argument("--no-check-nan", dest="check_nan", action="store_false")
    p.set_defaults(check_nan=True)
    p.add_argument("--check-every", type=int, default=50, help="Check finiteness every N steps (0 disables).")

    # IC sampling knobs
    p.add_argument("--ic-slope", type=float, default=2.5, help="Power-law slope for IC spectral filter.")
    p.add_argument("--ic-kmin", type=int, default=1, help="Minimum integer mode in IC band.")
    p.add_argument("--ic-kmax", type=int, default=10, help="Maximum integer mode in IC band.")
    p.add_argument("--ic-rms", type=float, default=5.0, help="Target RMS of initial vorticity ω0.")

    # Forcing knobs
    p.add_argument(
        "--forcing",
        type=str,
        default="kolmogorov",
        choices=["none", "kolmogorov", "spectral_band"],
        help="Forcing type in vorticity equation.",
    )
    p.add_argument("--forcing-amp", type=float, default=0.1, help="Forcing amplitude (interpretation depends on type).")
    p.add_argument("--forcing-k", type=int, default=1, help="Kolmogorov forcing wavenumber k.")
    p.add_argument("--forcing-band-kmin", type=int, default=3, help="spectral_band forcing min mode.")
    p.add_argument("--forcing-band-kmax", type=int, default=5, help="spectral_band forcing max mode.")
    p.add_argument(
        "--forcing-per-sample",
        action="store_true",
        help="If set, generate forcing per sample (else fixed forcing across dataset).",
    )

    # Optional observational corruption
    p.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise std added to selected fields (off if 0).")
    p.add_argument(
        "--noise-target",
        type=str,
        default="none",
        choices=["none", "omega0", "omegaT", "both"],
        help="Where to add noise.",
    )
    p.add_argument("--subsample-input", type=int, default=1, help="Subsample input omega0 by this stride (default 1).")
    p.add_argument("--subsample-output", type=int, default=1, help="Subsample output omegaT by this stride (default 1).")

    # Optional stored fields
    p.add_argument("--store-velocity", action="store_true", help="Store final velocity uT as (2,Nout,Nout).")
    p.add_argument("--store-params", action="store_true", help="Store scalar params (nu, dt, T, forcing info) per sample.")
    p.add_argument("--store-diagnostics", action="store_true", help="Store solver diagnostics scalars per sample.")
    p.add_argument("--store-sample-id", action="store_true", help="Store deterministic integer sample_id per example.")

    # Writing / progress
    p.add_argument(
        "--writer-batch-size",
        type=int,
        default=16,
        help="HF datasets writer batch size (smaller = lower memory, more overhead).",
    )
    p.add_argument("--progress-every", type=int, default=25, help="Print progress every N samples per split (0 disables).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = DatasetBuildConfig(
        dataset_name=args.dataset_name,
        output_root=args.output_root,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        seed=args.seed,
        dtype=args.dtype,
        N=args.N,
        L=args.L,
        T=args.T,
        dt=args.dt,
        nu=args.nu,
        integrator=args.integrator,
        dealias=args.dealias,
        adjust_dt_to_T=args.adjust_dt_to_T,
        dt_auto=args.dt_auto,
        cfl=args.cfl,
        ic_slope=args.ic_slope,
        ic_kmin=args.ic_kmin,
        ic_kmax=args.ic_kmax,
        ic_rms=args.ic_rms,
        forcing=args.forcing,
        forcing_amp=args.forcing_amp,
        forcing_k=args.forcing_k,
        forcing_band_kmin=args.forcing_band_kmin,
        forcing_band_kmax=args.forcing_band_kmax,
        forcing_per_sample=args.forcing_per_sample,
        store_velocity=args.store_velocity,
        store_params=args.store_params,
        store_diagnostics=args.store_diagnostics,
        store_sample_id=args.store_sample_id,
        noise_std=args.noise_std,
        noise_target=args.noise_target,
        subsample_input=args.subsample_input,
        subsample_output=args.subsample_output,
        check_nan=args.check_nan,
        check_every=args.check_every,
        writer_batch_size=args.writer_batch_size,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
    )

    os.makedirs(cfg.output_root, exist_ok=True)
    out_dir = build_and_save_dataset(cfg)
    print(f"Saved dataset to: {out_dir}")
    print(f"Metadata: {os.path.join(out_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
