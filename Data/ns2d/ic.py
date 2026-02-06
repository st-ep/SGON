# --- File: ns2d/ic.py ---
from __future__ import annotations

import numpy as np

from .spectral import rfft2, irfft2


def sample_vorticity_grf(
    rng: np.random.Generator,
    N: int,
    L: float,
    kx_int: np.ndarray,
    ky_int: np.ndarray,
    slope: float = 2.5,
    kmin: int = 1,
    kmax: int = 10,
    rms: float = 5.0,
) -> np.ndarray:
    """
    Sample a smooth periodic random initial vorticity field ω0(x,y) on [0,L)^2
    using a Gaussian random field constructed by spectral filtering of white noise.

    Construction:
      eta ~ N(0,1) i.i.d. in physical space
      eta_hat = rfft2(eta)
      omega_hat = eta_hat * filter(|k|)
      omega = irfft2(omega_hat)
      scale omega to have target RMS

    Filter:
      filter(k) = k^{-slope} for kmin <= k <= kmax, else 0
      with filter(0)=0

    Parameters
    ----------
    rng : numpy Generator
    N, L : grid resolution and domain size
    kx_int, ky_int : integer mode grids broadcastable to (N, N//2+1)
    slope : spectral slope exponent (>0 gives smoother fields)
    kmin, kmax : integer mode band limits (in box-mode units)
    rms : target root-mean-square of ω0

    Returns
    -------
    omega0 : (N,N) float64 array (caller can cast to float32 for storage)
    """
    if kmin < 0:
        raise ValueError("kmin must be >= 0")
    if kmax < kmin:
        raise ValueError("kmax must be >= kmin")
    if kmax > (N // 2):
        raise ValueError(f"kmax must be <= N/2 (got kmax={kmax}, N={N})")
    if rms <= 0:
        raise ValueError("rms must be > 0")

    eta = rng.standard_normal(size=(N, N), dtype=np.float64)
    eta_hat = rfft2(eta)

    k_mag = np.sqrt(kx_int.astype(np.float64) ** 2 + ky_int.astype(np.float64) ** 2)
    filt = np.zeros_like(k_mag, dtype=np.float64)

    band = (k_mag >= float(kmin)) & (k_mag <= float(kmax)) & (k_mag > 0.0)
    # Power-law scaling on integer mode magnitude (box modes)
    filt[band] = (k_mag[band] ** (-slope))

    omega_hat = eta_hat * filt
    # enforce zero mean vorticity
    omega_hat[0, 0] = 0.0 + 0.0j

    omega = irfft2(omega_hat, s=(N, N))

    # scale to target RMS
    cur_rms = float(np.sqrt(np.mean(omega**2)))
    if cur_rms < 1e-14:
        # extremely unlikely; resample
        omega = rng.standard_normal(size=(N, N), dtype=np.float64)
        cur_rms = float(np.sqrt(np.mean(omega**2)))
    omega *= (rms / cur_rms)

    return omega
