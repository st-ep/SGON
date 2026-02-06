# --- File: ns2d/solver.py ---
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .spectral import (
    irfft2,
    make_dealias_mask_rfft2,
    make_wavenumbers_rfft2,
    rfft2,
    spectral_poisson_solve_streamfunction,
    velocity_from_streamfunction_hat,
)


@dataclass(frozen=True)
class SolverConfig:
    N: int
    L: float
    T: float
    dt: float
    nu: float
    integrator: str = "cnab2"  # "cnab2" (fast, stable) or "etdrk4" (accurate)
    dealias: bool = True
    adjust_dt_to_T: bool = True  # make dt divide T exactly (recommended for CNAB2)
    check_nan: bool = True
    check_every: int = 50  # check finiteness every this many steps
    # Optional stability helper (computed once per sample, constant dt):
    dt_auto: bool = False
    cfl: float = 0.5


@dataclass
class SolverDiagnostics:
    n_steps: int
    dt_used: float
    max_abs_omega0: float
    max_abs_omegaT: float
    ok: bool
    nan_detected: bool


class NavierStokes2DVorticitySolver:
    """
    Pseudo-spectral 2D incompressible Navier–Stokes solver on a periodic box [0,L)^2,
    vorticity-streamfunction formulation:

        ω_t + u·∇ω = ν Δω + f
        Δψ = ω,   u = ∇⊥ψ = (∂ψ/∂y, -∂ψ/∂x)

    Spatial discretization:
      - Fourier pseudo-spectral using numpy rfft2/irfft2
      - 2/3-rule dealiasing applied to nonlinear term (and optionally to ω_hat)

    Time integration:
      - CNAB2: Crank–Nicolson for diffusion + Adams–Bashforth 2 for nonlinear+forcing (default; efficient)
      - ETDRK4: Exponential time-differencing RK4 (more accurate, heavier)
    """

    def __init__(self, N: int, L: float, dealias: bool = True):
        self.N = int(N)
        self.L = float(L)
        self.kx, self.ky, self.k2, self.kx_int, self.ky_int = make_wavenumbers_rfft2(self.N, self.L)
        self.dealias = bool(dealias)
        self.dealias_mask = make_dealias_mask_rfft2(self.N) if self.dealias else None

        # Workspace buffers (allocated lazily)
        self._psi_hat = None

    def _nonlinear_term_hat(self, omega_hat: np.ndarray, forcing_hat: np.ndarray) -> np.ndarray:
        """
        Compute N_hat(ω) = -FFT(u·∇ω) + forcing_hat, with dealiasing on the nonlinear term.

        Returns
        -------
        N_hat : (N, N//2+1) complex128
        """
        N = self.N

        psi_hat = spectral_poisson_solve_streamfunction(omega_hat, self.k2)
        u, v = velocity_from_streamfunction_hat(psi_hat, self.kx, self.ky, N)

        omega_x = irfft2(1j * self.kx * omega_hat, s=(N, N))
        omega_y = irfft2(1j * self.ky * omega_hat, s=(N, N))

        adv = u * omega_x + v * omega_y
        adv_hat = rfft2(adv)

        N_hat = -adv_hat + forcing_hat

        if self.dealias_mask is not None:
            N_hat = N_hat * self.dealias_mask

        return N_hat

    def _maybe_check_finite(self, arr: np.ndarray) -> bool:
        return bool(np.isfinite(arr).all())

    def _estimate_umax(self, omega_hat: np.ndarray) -> float:
        """
        Estimate max speed |u|_∞ from ω_hat.
        Used only for optional dt_auto (computed once per sample).
        """
        psi_hat = spectral_poisson_solve_streamfunction(omega_hat, self.k2)
        u, v = velocity_from_streamfunction_hat(psi_hat, self.kx, self.ky, self.N)
        speed = np.sqrt(u * u + v * v)
        return float(np.max(speed))

    def integrate(
        self,
        omega0: np.ndarray,
        cfg: SolverConfig,
        forcing_hat: np.ndarray,
        return_velocity: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None, SolverDiagnostics]:
        """
        Integrate from t=0 to t=T given initial vorticity omega0.

        Parameters
        ----------
        omega0 : (N,N) array-like
            Initial vorticity field in physical space.
        cfg : SolverConfig
            Contains dt, nu, T, integrator, etc.
        forcing_hat : (N, N//2+1) complex
            Time-independent forcing in Fourier space.
        return_velocity : bool
            If True, also return (uT, vT) at final time.

        Returns
        -------
        omegaT : (N,N) float64 array
        (uT, vT) : optional, (N,N) float64 arrays
        diagnostics : SolverDiagnostics
        """
        N = self.N
        if omega0.shape != (N, N):
            raise ValueError(f"omega0 must have shape ({N},{N}), got {omega0.shape}")

        if cfg.T <= 0:
            raise ValueError("T must be > 0")
        if cfg.dt <= 0:
            raise ValueError("dt must be > 0")
        if cfg.nu < 0:
            raise ValueError("nu must be >= 0")

        omega_hat = rfft2(omega0.astype(np.float64, copy=False))

        # Optional: constant dt computed once from initial CFL.
        dt_requested = float(cfg.dt)
        if cfg.dt_auto:
            dx = cfg.L / float(cfg.N)
            umax0 = self._estimate_umax(omega_hat)
            # avoid division by zero
            dt_cfl = float(cfg.cfl) * dx / max(umax0, 1e-12)
            dt_requested = min(dt_requested, dt_cfl)

        # Use constant dt; optionally adjust so that T is hit exactly with integer steps.
        if cfg.adjust_dt_to_T:
            n_steps = int(math.ceil(cfg.T / dt_requested))
            dt_used = float(cfg.T) / float(n_steps)
        else:
            # For stability/consistency, still use integer steps and take a smaller last step
            # (CNAB2 assumes constant dt; so we still keep constant dt and may overshoot slightly).
            n_steps = int(math.floor(cfg.T / dt_requested))
            if n_steps < 1:
                n_steps = 1
            dt_used = dt_requested
            # Note: if adjust_dt_to_T=False, final time is ~n_steps*dt_used (<=T).
            # We intentionally keep CNAB2 constant-step; user can set adjust_dt_to_T=True (recommended).
            # We'll treat "T_effective" as n_steps*dt_used in metadata? Here we keep cfg.T in cfg.
        # Effective final time when adjust_dt_to_T=False:
        T_effective = float(cfg.T) if cfg.adjust_dt_to_T else float(n_steps * dt_used)

        max_abs_omega0 = float(np.max(np.abs(omega0)))

        nan_detected = False
        ok = True

        integrator = cfg.integrator.lower().strip()
        if integrator == "cnab2":
            omega_hat, nan_detected = self._integrate_cnab2(
                omega_hat=omega_hat,
                forcing_hat=forcing_hat,
                nu=float(cfg.nu),
                dt=dt_used,
                n_steps=n_steps,
                check_nan=cfg.check_nan,
                check_every=int(cfg.check_every),
            )
        elif integrator == "etdrk4":
            omega_hat, nan_detected = self._integrate_etdrk4(
                omega_hat=omega_hat,
                forcing_hat=forcing_hat,
                nu=float(cfg.nu),
                dt=dt_used,
                n_steps=n_steps,
                check_nan=cfg.check_nan,
                check_every=int(cfg.check_every),
            )
        else:
            raise ValueError(f"Unknown integrator: {cfg.integrator}")

        if nan_detected:
            ok = False

        omegaT = irfft2(omega_hat, s=(N, N))
        max_abs_omegaT = float(np.max(np.abs(omegaT)))

        if cfg.check_nan and (not np.isfinite(omegaT).all()):
            nan_detected = True
            ok = False

        vel_out = None
        if return_velocity:
            psi_hat = spectral_poisson_solve_streamfunction(omega_hat, self.k2)
            uT, vT = velocity_from_streamfunction_hat(psi_hat, self.kx, self.ky, N)
            vel_out = (uT, vT)

        diagnostics = SolverDiagnostics(
            n_steps=int(n_steps),
            dt_used=float(dt_used),
            max_abs_omega0=max_abs_omega0,
            max_abs_omegaT=max_abs_omegaT,
            ok=bool(ok),
            nan_detected=bool(nan_detected),
        )
        # If adjust_dt_to_T=False, user probably expects T_effective < T; keep it in mind externally.
        # We keep cfg.T as "requested" and dt_used + n_steps defines actual evolution time.
        # The generator records dt_used and n_steps in metadata per-split summary.
        return omegaT, vel_out, diagnostics

    def _integrate_cnab2(
        self,
        omega_hat: np.ndarray,
        forcing_hat: np.ndarray,
        nu: float,
        dt: float,
        n_steps: int,
        check_nan: bool,
        check_every: int,
    ) -> tuple[np.ndarray, bool]:
        """
        CNAB2 (semi-implicit) time stepping in Fourier space:
          (ω^{n+1} - ω^n)/dt = -0.5 ν k^2 (ω^{n+1}+ω^n) + AB2(N^n)
        where N^n = -FFT(u·∇ω)^n + f_hat

        Initialization: first step uses CN for diffusion + forward Euler for N.
        """
        # Linear operator for diffusion
        k2 = self.k2
        a = 1.0 - 0.5 * dt * nu * k2
        b = 1.0 + 0.5 * dt * nu * k2

        nan_detected = False

        # Compute N^0
        N0 = self._nonlinear_term_hat(omega_hat, forcing_hat)

        # First step (CN + Euler)
        omega_hat_1 = (a * omega_hat + dt * N0) / b
        if self.dealias_mask is not None:
            omega_hat_1 = omega_hat_1 * self.dealias_mask

        if check_nan and (not self._maybe_check_finite(omega_hat_1)):
            return omega_hat_1, True

        Nm1 = N0
        omega_hat_n = omega_hat_1

        # Main CNAB2 loop
        for n in range(1, n_steps):
            Nn = self._nonlinear_term_hat(omega_hat_n, forcing_hat)
            omega_hat_np1 = (a * omega_hat_n + dt * (1.5 * Nn - 0.5 * Nm1)) / b
            if self.dealias_mask is not None:
                omega_hat_np1 = omega_hat_np1 * self.dealias_mask

            if check_nan and (check_every > 0) and (n % check_every == 0):
                if not self._maybe_check_finite(omega_hat_np1):
                    nan_detected = True
                    omega_hat_n = omega_hat_np1
                    break

            Nm1 = Nn
            omega_hat_n = omega_hat_np1

        return omega_hat_n, nan_detected

    def _etdrk4_coeffs(self, L: np.ndarray, dt: float, M: int = 16) -> tuple[np.ndarray, ...]:
        """
        Precompute ETDRK4 coefficients for diagonal linear operator L (same shape as omega_hat).

        Uses the Kassam–Trefethen contour integral approach with M complex points.

        Returns
        -------
        E, E2, Q, f1, f2, f3 : arrays broadcastable to omega_hat shape
        """
        E = np.exp(dt * L)
        E2 = np.exp(dt * L / 2.0)

        # Complex roots of unity on a shifted contour
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)  # (M,)
        # Broadcast: (..., M)
        LR = dt * L[..., None] + r[None, None, :]

        # contour averages
        Q = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=-1)
        f1 = dt * np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR * LR)) / (LR**3), axis=-1)
        f2 = dt * np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR**3), axis=-1)
        f3 = dt * np.mean((-4.0 - 3.0 * LR - LR * LR + np.exp(LR) * (4.0 - LR)) / (LR**3), axis=-1)
        return E, E2, Q, f1, f2, f3

    def _integrate_etdrk4(
        self,
        omega_hat: np.ndarray,
        forcing_hat: np.ndarray,
        nu: float,
        dt: float,
        n_steps: int,
        check_nan: bool,
        check_every: int,
    ) -> tuple[np.ndarray, bool]:
        """
        ETDRK4 time stepping for:
          ω_t = L ω + N(ω),
        where L = -ν k^2 (diagonal) and N(ω) = -FFT(u·∇ω) + forcing_hat.
        """
        L = -nu * self.k2  # diagonal linear operator
        E, E2, Q, f1, f2, f3 = self._etdrk4_coeffs(L, dt, M=16)

        nan_detected = False
        w = omega_hat.astype(np.complex128, copy=False)

        for n in range(n_steps):
            Nv = self._nonlinear_term_hat(w, forcing_hat)
            a = E2 * w + Q * Nv
            Na = self._nonlinear_term_hat(a, forcing_hat)
            b = E2 * w + Q * Na
            Nb = self._nonlinear_term_hat(b, forcing_hat)
            c = E2 * a + Q * (2.0 * Nb - Nv)
            Nc = self._nonlinear_term_hat(c, forcing_hat)

            w = E * w + f1 * Nv + f2 * (Na + Nb) + f3 * Nc

            if self.dealias_mask is not None:
                w = w * self.dealias_mask

            if check_nan and (check_every > 0) and (n % check_every == 0):
                if not self._maybe_check_finite(w):
                    nan_detected = True
                    break

        return w, nan_detected
