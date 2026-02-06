# --- File: ns2d/dataset_writer.py ---
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import datasets

from .forcing import make_forcing_vorticity_hat
from .ic import sample_vorticity_grf
from .solver import NavierStokes2DVorticitySolver, SolverConfig


@dataclass
class DatasetBuildConfig:
    # Dataset structure
    dataset_name: str
    output_root: str
    num_samples: int
    train_ratio: float
    seed: int
    dtype: str = "float32"  # stored dtype

    # Physics & numerics
    N: int = 64
    L: float = 1.0
    T: float = 1.0
    dt: float = 2e-3
    nu: float = 1e-3
    integrator: str = "cnab2"
    dealias: bool = True
    adjust_dt_to_T: bool = True
    dt_auto: bool = False
    cfl: float = 0.5

    # IC sampling
    ic_slope: float = 2.5
    ic_kmin: int = 1
    ic_kmax: int = 10
    ic_rms: float = 5.0

    # Forcing
    forcing: str = "kolmogorov"
    forcing_amp: float = 0.1
    forcing_k: int = 1
    forcing_band_kmin: int = 3
    forcing_band_kmax: int = 5
    forcing_seed_offset: int = 12345  # global seed offset for random forcing kinds
    forcing_per_sample: bool = False

    # Optional dataset extras
    store_velocity: bool = False
    store_params: bool = False
    store_diagnostics: bool = False
    store_sample_id: bool = False

    # Noise / subsampling (off by default)
    noise_std: float = 0.0
    noise_target: str = "none"  # "none", "omega0", "omegaT", "both"
    subsample_input: int = 1
    subsample_output: int = 1

    # Fail-fast / logging
    check_nan: bool = True
    check_every: int = 50
    progress_every: int = 25

    # HF datasets writing
    writer_batch_size: int = 16
    overwrite: bool = False


class RunningStats:
    """Simple online stats collector for scalar diagnostics."""
    def __init__(self) -> None:
        self.n = 0
        self.sums: Dict[str, float] = {}
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}

    def update(self, **vals: float) -> None:
        self.n += 1
        for k, v in vals.items():
            v = float(v)
            self.sums[k] = self.sums.get(k, 0.0) + v
            self.mins[k] = v if k not in self.mins else min(self.mins[k], v)
            self.maxs[k] = v if k not in self.maxs else max(self.maxs[k], v)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"count": self.n, "mean": {}, "min": {}, "max": {}}
        if self.n == 0:
            return out
        for k, s in self.sums.items():
            out["mean"][k] = s / float(self.n)
        out["min"] = dict(self.mins)
        out["max"] = dict(self.maxs)
        return out


def _ensure_empty_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path} (use --overwrite to replace)")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _dtype_to_hf(dtype: str) -> str:
    dtype = dtype.lower().strip()
    if dtype not in ("float32", "float64"):
        raise ValueError(f"Unsupported dtype for storage: {dtype}")
    return dtype


def _make_features(cfg: DatasetBuildConfig, Nin: int, Nout: int) -> datasets.Features:
    dtype = _dtype_to_hf(cfg.dtype)
    feats: Dict[str, Any] = {
        "omega0": datasets.Array2D(shape=(Nin, Nin), dtype=dtype),
        "omegaT": datasets.Array2D(shape=(Nout, Nout), dtype=dtype),
    }
    if cfg.store_velocity:
        feats["uT"] = datasets.Array3D(shape=(2, Nout, Nout), dtype=dtype)  # [u, v]
    if cfg.store_params:
        feats["nu"] = datasets.Value("float32")
        feats["T"] = datasets.Value("float32")
        feats["dt"] = datasets.Value("float32")
        feats["forcing_kind"] = datasets.Value("string")
        feats["forcing_amp"] = datasets.Value("float32")
        feats["forcing_k"] = datasets.Value("int32")
    if cfg.store_diagnostics:
        feats["n_steps"] = datasets.Value("int32")
        feats["dt_used"] = datasets.Value("float32")
        feats["max_abs_omega0"] = datasets.Value("float32")
        feats["max_abs_omegaT"] = datasets.Value("float32")
        feats["ok"] = datasets.Value("bool")
        feats["nan_detected"] = datasets.Value("bool")
    if cfg.store_sample_id:
        feats["sample_id"] = datasets.Value("int32")
    return datasets.Features(feats)


def _apply_subsample(field: np.ndarray, s: int) -> np.ndarray:
    if s == 1:
        return field
    return field[::s, ::s]


def _apply_noise(rng: np.random.Generator, field: np.ndarray, std: float) -> np.ndarray:
    if std <= 0.0:
        return field
    return field + rng.normal(loc=0.0, scale=std, size=field.shape).astype(field.dtype, copy=False)


def _make_sample_rng(global_seed: int, sample_id: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(global_seed), int(sample_id)])
    return np.random.default_rng(ss)


def _build_split_indices(num_samples: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(num_samples).tolist()
    n_train = int(round(train_ratio * num_samples))
    n_train = max(1, min(num_samples - 1, n_train))
    train_ids = perm[:n_train]
    test_ids = perm[n_train:]
    return train_ids, test_ids


def make_pde_description() -> str:
    return (
        "2D incompressible Navier–Stokes on a periodic domain (2-torus), vorticity–streamfunction form:\n"
        "  ω_t + u·∇ω = ν Δω + f\n"
        "  Δψ = ω,  u = ∇⊥ψ = (∂ψ/∂y, -∂ψ/∂x)\n"
        "Spatial discretization: Fourier pseudo-spectral (numpy rfft2/irfft2), 2/3-rule dealiasing on nonlinear term.\n"
        "Time integration: configurable (CNAB2 or ETDRK4)."
    )


def build_and_save_dataset(cfg: DatasetBuildConfig) -> str:
    """
    Build HF DatasetDict with train/test splits and save to disk under:
        <output_root>/<dataset_name>/

    Returns
    -------
    out_dir : str
        Path to saved dataset directory.
    """
    out_dir = os.path.join(cfg.output_root, cfg.dataset_name)
    _ensure_empty_dir(out_dir, overwrite=cfg.overwrite)

    if cfg.N <= 0:
        raise ValueError("N must be > 0")
    if cfg.L <= 0:
        raise ValueError("L must be > 0")
    if cfg.num_samples <= 1:
        raise ValueError("num_samples must be > 1")
    if cfg.subsample_input < 1 or cfg.subsample_output < 1:
        raise ValueError("subsample factors must be >= 1")
    if cfg.N % cfg.subsample_input != 0:
        raise ValueError("N must be divisible by subsample_input")
    if cfg.N % cfg.subsample_output != 0:
        raise ValueError("N must be divisible by subsample_output")

    Nin = cfg.N // cfg.subsample_input
    Nout = cfg.N // cfg.subsample_output

    # Common solver grid/FFT setup (nu can still be per-sample, but grids are fixed)
    solver = NavierStokes2DVorticitySolver(N=cfg.N, L=cfg.L, dealias=cfg.dealias)

    # Global forcing (default): fixed across dataset for standard benchmark behavior.
    forcing_rng_global = np.random.default_rng(int(cfg.seed) + int(cfg.forcing_seed_offset))
    forcing_hat_global = make_forcing_vorticity_hat(
        kind=cfg.forcing,
        N=cfg.N,
        L=cfg.L,
        kx_int=solver.kx_int,
        ky_int=solver.ky_int,
        amp=float(cfg.forcing_amp),
        k=int(cfg.forcing_k),
        band_kmin=int(cfg.forcing_band_kmin),
        band_kmax=int(cfg.forcing_band_kmax),
        rng=forcing_rng_global,
    ).astype(np.complex128, copy=False)

    train_ids, test_ids = _build_split_indices(cfg.num_samples, cfg.train_ratio, cfg.seed)

    stats_train = RunningStats()
    stats_test = RunningStats()

    features = _make_features(cfg, Nin=Nin, Nout=Nout)

    def make_generator(split_name: str, sample_ids: list[int], stats: RunningStats) -> Iterator[Dict[str, Any]]:
        for i, sid in enumerate(sample_ids):
            rng = _make_sample_rng(cfg.seed, sid)

            # Per-sample forcing (optional)
            if cfg.forcing_per_sample:
                forcing_hat = make_forcing_vorticity_hat(
                    kind=cfg.forcing,
                    N=cfg.N,
                    L=cfg.L,
                    kx_int=solver.kx_int,
                    ky_int=solver.ky_int,
                    amp=float(cfg.forcing_amp),
                    k=int(cfg.forcing_k),
                    band_kmin=int(cfg.forcing_band_kmin),
                    band_kmax=int(cfg.forcing_band_kmax),
                    rng=rng,
                ).astype(np.complex128, copy=False)
            else:
                forcing_hat = forcing_hat_global

            # Initial condition
            omega0_full = sample_vorticity_grf(
                rng=rng,
                N=cfg.N,
                L=cfg.L,
                kx_int=solver.kx_int,
                ky_int=solver.ky_int,
                slope=float(cfg.ic_slope),
                kmin=int(cfg.ic_kmin),
                kmax=int(cfg.ic_kmax),
                rms=float(cfg.ic_rms),
            )

            # Optional noise on input
            if cfg.noise_target in ("omega0", "both") and cfg.noise_std > 0:
                omega0_full = _apply_noise(rng, omega0_full.astype(np.float64, copy=False), float(cfg.noise_std))

            # Solve
            solver_cfg = SolverConfig(
                N=int(cfg.N),
                L=float(cfg.L),
                T=float(cfg.T),
                dt=float(cfg.dt),
                nu=float(cfg.nu),
                integrator=str(cfg.integrator),
                dealias=bool(cfg.dealias),
                adjust_dt_to_T=bool(cfg.adjust_dt_to_T),
                check_nan=bool(cfg.check_nan),
                check_every=int(cfg.check_every),
                dt_auto=bool(cfg.dt_auto),
                cfl=float(cfg.cfl),
            )

            omegaT_full, velT, diag = solver.integrate(
                omega0=omega0_full,
                cfg=solver_cfg,
                forcing_hat=forcing_hat,
                return_velocity=bool(cfg.store_velocity),
            )

            if not diag.ok:
                # Fail-fast: abort dataset creation if any sample blows up.
                raise RuntimeError(
                    f"Solver failure (split={split_name}, sample_id={sid}): "
                    f"nan_detected={diag.nan_detected}, max|ω0|={diag.max_abs_omega0:.3e}, max|ωT|={diag.max_abs_omegaT:.3e}"
                )

            # Optional noise on output
            if cfg.noise_target in ("omegaT", "both") and cfg.noise_std > 0:
                omegaT_full = _apply_noise(rng, omegaT_full.astype(np.float64, copy=False), float(cfg.noise_std))
                # (If noise makes NaNs, fail-fast)
                if cfg.check_nan and (not np.isfinite(omegaT_full).all()):
                    raise RuntimeError(f"Non-finite omegaT after noise (split={split_name}, sample_id={sid})")

            # Subsample
            omega0 = _apply_subsample(omega0_full, int(cfg.subsample_input))
            omegaT = _apply_subsample(omegaT_full, int(cfg.subsample_output))

            # Cast for storage
            np_dtype = np.float32 if cfg.dtype == "float32" else np.float64
            omega0 = omega0.astype(np_dtype, copy=False)
            omegaT = omegaT.astype(np_dtype, copy=False)

            ex: Dict[str, Any] = {
                "omega0": omega0,
                "omegaT": omegaT,
            }

            if cfg.store_velocity and velT is not None:
                uT_full, vT_full = velT
                uT = _apply_subsample(uT_full, int(cfg.subsample_output)).astype(np_dtype, copy=False)
                vT = _apply_subsample(vT_full, int(cfg.subsample_output)).astype(np_dtype, copy=False)
                ex["uT"] = np.stack([uT, vT], axis=0)

            if cfg.store_params:
                ex["nu"] = np.float32(cfg.nu)
                ex["T"] = np.float32(cfg.T)
                ex["dt"] = np.float32(cfg.dt)
                ex["forcing_kind"] = str(cfg.forcing)
                ex["forcing_amp"] = np.float32(cfg.forcing_amp)
                ex["forcing_k"] = np.int32(cfg.forcing_k)

            if cfg.store_diagnostics:
                ex["n_steps"] = np.int32(diag.n_steps)
                ex["dt_used"] = np.float32(diag.dt_used)
                ex["max_abs_omega0"] = np.float32(diag.max_abs_omega0)
                ex["max_abs_omegaT"] = np.float32(diag.max_abs_omegaT)
                ex["ok"] = bool(diag.ok)
                ex["nan_detected"] = bool(diag.nan_detected)

            if cfg.store_sample_id:
                ex["sample_id"] = np.int32(sid)

            stats.update(
                n_steps=float(diag.n_steps),
                dt_used=float(diag.dt_used),
                max_abs_omega0=float(diag.max_abs_omega0),
                max_abs_omegaT=float(diag.max_abs_omegaT),
            )

            if cfg.progress_every > 0 and ((i + 1) % int(cfg.progress_every) == 0 or (i + 1) == len(sample_ids)):
                print(f"[{split_name}] generated {i+1}/{len(sample_ids)} samples")

            yield ex

    train_ds = datasets.Dataset.from_generator(
        make_generator,
        gen_kwargs={"split_name": "train", "sample_ids": train_ids, "stats": stats_train},
        features=features,
        writer_batch_size=int(cfg.writer_batch_size),
        cache_dir=os.path.join(out_dir, "_hf_cache"),
        keep_in_memory=False,
    )
    test_ds = datasets.Dataset.from_generator(
        make_generator,
        gen_kwargs={"split_name": "test", "sample_ids": test_ids, "stats": stats_test},
        features=features,
        writer_batch_size=int(cfg.writer_batch_size),
        cache_dir=os.path.join(out_dir, "_hf_cache"),
        keep_in_memory=False,
    )

    dsdict = datasets.DatasetDict({"train": train_ds, "test": test_ds})
    dsdict.save_to_disk(out_dir)

    # Metadata
    meta = {
        "pde": make_pde_description(),
        "axis_ordering": {
            "omega": "omega[i, j] corresponds to x = i*(L/N), y = j*(L/N) (ij indexing).",
            "uT": "uT[0, i, j] = u(x,y), uT[1, i, j] = v(x,y) if stored.",
        },
        "args": cfg.__dict__.copy(),
        "grid": {"N": cfg.N, "L": cfg.L, "dx": cfg.L / float(cfg.N)},
        "dataset": {
            "num_samples_total": cfg.num_samples,
            "train_samples": len(train_ids),
            "test_samples": len(test_ids),
            "train_ratio": cfg.train_ratio,
            "stored_dtype": cfg.dtype,
            "input_shape": [Nin, Nin],
            "output_shape": [Nout, Nout],
            "fields": {
                "omega0": True,
                "omegaT": True,
                "uT": bool(cfg.store_velocity),
            },
        },
        "forcing": {
            "kind": cfg.forcing,
            "amp": cfg.forcing_amp,
            "k": cfg.forcing_k,
            "band_kmin": cfg.forcing_band_kmin,
            "band_kmax": cfg.forcing_band_kmax,
            "per_sample": bool(cfg.forcing_per_sample),
        },
        "ic_sampling": {
            "type": "grf_spectral_filter",
            "slope": cfg.ic_slope,
            "kmin": cfg.ic_kmin,
            "kmax": cfg.ic_kmax,
            "rms": cfg.ic_rms,
        },
        "solver": {
            "integrator": cfg.integrator,
            "dealiasing": "2/3-rule" if cfg.dealias else "off",
            "dt_requested": cfg.dt,
            "T_requested": cfg.T,
            "adjust_dt_to_T": bool(cfg.adjust_dt_to_T),
            "dt_auto": bool(cfg.dt_auto),
            "cfl": cfg.cfl,
            "nu": cfg.nu,
        },
        "split_diagnostics_summary": {
            "train": stats_train.summary(),
            "test": stats_test.summary(),
        },
    }

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return out_dir
