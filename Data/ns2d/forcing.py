from __future__ import annotations

from typing import Optional

import numpy as np

from .spectral import irfft2, rfft2


def _validate_band(band_kmin: int, band_kmax: int, N: int) -> None:
    if band_kmin < 0:
        raise ValueError("band_kmin must be >= 0")
    if band_kmax < band_kmin:
        raise ValueError("band_kmax must be >= band_kmin")
    if band_kmax > (N // 2):
        raise ValueError(f"band_kmax must be <= N/2 (got band_kmax={band_kmax}, N={N})")


def _scale_hat_to_rms(
    forcing_hat: np.ndarray, target_rms: float, N: int
) -> np.ndarray:
    if target_rms <= 0.0:
        return np.zeros_like(forcing_hat)
    forcing = irfft2(forcing_hat, s=(N, N))
    cur_rms = float(np.sqrt(np.mean(forcing**2)))
    if cur_rms < 1e-14:
        return np.zeros_like(forcing_hat)
    return forcing_hat * (float(target_rms) / cur_rms)


def make_forcing_vorticity_hat(
    kind: str,
    N: int,
    L: float,
    kx_int: np.ndarray,
    ky_int: np.ndarray,
    amp: float = 0.1,
    k: int = 1,
    band_kmin: int = 3,
    band_kmax: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Construct time-independent vorticity forcing in Fourier space (rfft2 layout).

    Parameters
    ----------
    kind : {"none", "kolmogorov", "spectral_band"}
        Forcing type.
    N, L : int, float
        Grid size and domain length.
    kx_int, ky_int : arrays
        Integer mode grids (from make_wavenumbers_rfft2), broadcastable to (N, N//2+1).
    amp : float
        Forcing amplitude. Interpretation depends on kind:
        - kolmogorov: physical-space peak amplitude of the sinusoid
        - spectral_band: target RMS amplitude in physical space
    k : int
        Kolmogorov forcing wavenumber.
    band_kmin, band_kmax : int
        Spectral-band forcing integer mode range (inclusive).
    rng : np.random.Generator, optional
        RNG used for random spectral-band forcing.
    """
    kind = str(kind).lower().strip()
    N = int(N)
    L = float(L)

    if kind == "none":
        return np.zeros((N, N // 2 + 1), dtype=np.complex128)

    if kind == "kolmogorov":
        if k <= 0:
            raise ValueError("k must be >= 1 for kolmogorov forcing")
        if k > (N // 2):
            raise ValueError(f"k must be <= N/2 (got k={k}, N={N})")
        y = np.linspace(0.0, L, N, endpoint=False)
        forcing_1d = float(amp) * np.sin(2.0 * np.pi * float(k) * y / L)
        forcing = np.broadcast_to(forcing_1d[None, :], (N, N))
        return rfft2(forcing)

    if kind == "spectral_band":
        _validate_band(int(band_kmin), int(band_kmax), N)
        if rng is None:
            rng = np.random.default_rng()

        noise = rng.standard_normal(size=(N, N))
        noise_hat = rfft2(noise)
        k_mag = np.sqrt(kx_int.astype(np.float64) ** 2 + ky_int.astype(np.float64) ** 2)
        band = (k_mag >= float(band_kmin)) & (k_mag <= float(band_kmax)) & (k_mag > 0.0)
        forcing_hat = noise_hat * band
        forcing_hat = _scale_hat_to_rms(forcing_hat, float(amp), N)
        return forcing_hat.astype(np.complex128, copy=False)

    raise ValueError(f"Unknown forcing kind: {kind}")
