# --- File: ns2d/spectral.py ---
from __future__ import annotations

import numpy as np


def make_grid(N: int, L: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Create periodic grid coordinates on [0, L) with N points per dimension.

    Returns
    -------
    x, y : (N,) arrays
        Coordinates in physical space.
    dx : float
        Grid spacing, dx = L / N.
    """
    dx = L / float(N)
    x = np.linspace(0.0, L, N, endpoint=False)
    y = np.linspace(0.0, L, N, endpoint=False)
    return x, y, dx


def make_wavenumbers_rfft2(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wavenumbers for numpy rfft2 layout:
      - axis 0 uses fftfreq (full)
      - axis 1 uses rfftfreq (non-negative)

    Returns
    -------
    kx, ky : arrays broadcastable to (N, N//2+1)
        Radian wavenumbers (2π * cycles/length).
        kx has shape (N, 1), ky has shape (1, N//2+1).
    k2 : (N, N//2+1) array
        k^2 = kx^2 + ky^2.
    kx_int, ky_int : arrays broadcastable to (N, N//2+1)
        Integer (mode index) wavenumbers corresponding to cycles per box:
        freq = m / L  =>  m = freq * L.
        Useful for spectral cutoff/slope decisions independent of L's scaling.
    """
    # cycles per unit length
    fx = np.fft.fftfreq(N, d=L / N)  # shape (N,)
    fy = np.fft.rfftfreq(N, d=L / N)  # shape (N//2+1,)

    # integer mode numbers
    # fx = m/L => m = fx*L (should be integer-valued up to fp rounding)
    kx_int = (fx * L).astype(np.int64)[:, None]  # (N,1)
    ky_int = (fy * L).astype(np.int64)[None, :]  # (1, N//2+1)

    # radian wavenumbers
    kx = (2.0 * np.pi * fx)[:, None]  # (N,1)
    ky = (2.0 * np.pi * fy)[None, :]  # (1, N//2+1)

    k2 = kx * kx + ky * ky
    return kx, ky, k2, kx_int, ky_int


def make_dealias_mask_rfft2(N: int) -> np.ndarray:
    """
    Standard 2/3-rule dealiasing mask for quadratic nonlinearity, in rfft2 layout.

    For even N, fftfreq integer modes are: 0,1,...,N/2-1,-N/2,...,-1
    2/3 rule keeps |k| <= floor(N/3).

    Returns
    -------
    mask : (N, N//2+1) boolean array
    """
    cutoff = N // 3  # floor(N/3)
    kx_int = (np.fft.fftfreq(N) * N).astype(np.int64)[:, None]        # (N,1)
    ky_int = (np.fft.rfftfreq(N) * N).astype(np.int64)[None, :]       # (1,N//2+1)
    mask = (np.abs(kx_int) <= cutoff) & (np.abs(ky_int) <= cutoff)
    return mask


def rfft2(a: np.ndarray) -> np.ndarray:
    """Wrapper for rfft2 with consistent axes."""
    return np.fft.rfft2(a, axes=(0, 1))


def irfft2(A: np.ndarray, s: tuple[int, int]) -> np.ndarray:
    """Wrapper for irfft2 with consistent axes."""
    return np.fft.irfft2(A, s=s, axes=(0, 1))


def spectral_poisson_solve_streamfunction(omega_hat: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """
    Solve Δψ = ω in Fourier space, i.e. -(k^2) ψ_hat = ω_hat => ψ_hat = -ω_hat / k^2,
    with ψ_hat(k=0)=0.

    Parameters
    ----------
    omega_hat : (N, N//2+1) complex array
    k2 : (N, N//2+1) float array

    Returns
    -------
    psi_hat : (N, N//2+1) complex array
    """
    psi_hat = np.zeros_like(omega_hat)
    # Avoid division by zero at k=0
    nonzero = k2 > 0.0
    psi_hat[nonzero] = -omega_hat[nonzero] / k2[nonzero]
    return psi_hat


def velocity_from_streamfunction_hat(
    psi_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    u = ∂ψ/∂y, v = -∂ψ/∂x.

    Returns u,v in physical space (real arrays shape (N,N)).
    """
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    u = irfft2(u_hat, s=(N, N))
    v = irfft2(v_hat, s=(N, N))
    return u, v
