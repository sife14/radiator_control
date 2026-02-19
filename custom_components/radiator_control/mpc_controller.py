"""
Model Predictive Controller (MPC)
=================================
Optimiert Temperatur-Offset über einen Prädiktionshorizont.

Verwendet nur numpy — kein cvxpy/osqp nötig.
Das QP wird durch Eliminierung der Temperaturzustände auf ein
box-constrained QP in den Steuergrößen reduziert und mit
Projected Gradient Descent gelöst.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .model import ThermalModel

logger = logging.getLogger(__name__)


@dataclass
class MPCConfig:
    """MPC Konfiguration."""

    horizon_steps: int = 20
    control_horizon: int = 10
    dt_minutes: float = 5.0
    weight_comfort: float = 1.0
    weight_energy: float = 0.1
    weight_smoothness: float = 0.05
    offset_min: float = -5.0
    offset_max: float = 5.0
    temp_min: float = 18.0
    temp_max: float = 26.0
    verbose: bool = False


@dataclass
class MPCResult:
    """Ergebnis einer MPC-Optimierung."""

    optimal_offset: float
    optimal_offsets: np.ndarray
    predicted_temps: np.ndarray
    cost_value: float
    solve_time_ms: float
    status: str

    def to_dict(self) -> Dict:
        return {
            "optimal_offset": self.optimal_offset,
            "optimal_offsets": self.optimal_offsets.tolist(),
            "predicted_temps": self.predicted_temps.tolist(),
            "cost_value": self.cost_value,
            "solve_time_ms": self.solve_time_ms,
            "status": self.status,
        }


class MPCController:
    """
    Model Predictive Controller für Heizungsregelung.

    Kostenfunktion:
    J = Σ [ w_c * (T - T_target)² + w_e * offset² + w_s * Δoffset² ]

    Lösung via Elimination der Temperaturzustände (linear dynamics)
    und Projected Gradient Descent auf dem reduzierten box-QP.
    """

    def __init__(
        self,
        model: ThermalModel,
        config: Optional[MPCConfig] = None,
    ):
        self.model = model
        self.config = config or MPCConfig()
        self.solve_count = 0
        self.total_solve_time = 0.0

    def _build_dynamics_matrices(
        self,
        outside_temps: np.ndarray,
        window_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Eliminate temperature states via linear dynamics.

        Returns s_init, S_u, d such that:
            T_pred[k] = s_init[k] * T_init + (S_u @ u)[k] + d[k]

        where T_pred has N elements (predicted temps at steps 1..N).
        """
        cfg = self.config
        p = self.model.params
        N = cfg.horizon_steps
        M = cfg.control_horizon
        dt = cfg.dt_minutes
        tau = max(p.tau, 10.0)

        # Per-step linearized coefficients:
        # T[k+1] = a[k]*T[k] + b[k]*u_k + c[k]
        a = np.empty(N)
        b = np.empty(N)
        c = np.empty(N)

        for k in range(N):
            coupling = p.k_outside + p.k_window * float(window_states[k])
            a[k] = 1.0 - dt / tau * coupling
            b[k] = -(dt / tau) * p.k_heater
            c[k] = (dt / tau) * coupling * float(outside_temps[k])

        # Build matrices by recursive expansion:
        # T[k+1] = a_prod[0..k] * T_init + Σ_j a_prod[j+1..k] * b[j] * u_j + Σ_j a_prod[j+1..k] * c[j]
        s_init = np.empty(N)
        S_u = np.zeros((N, M))
        d = np.empty(N)

        # Cumulative product of a coefficients
        a_cumprod = np.cumprod(a)  # a_cumprod[k] = prod(a[0:k+1])

        for k in range(N):
            s_init[k] = a_cumprod[k]
            d_k = 0.0
            for j in range(k + 1):
                # product of a[j+1..k]
                ratio = a_cumprod[k] / a_cumprod[j] if a_cumprod[j] != 0 else 0.0
                m = min(j, M - 1)  # control input index
                S_u[k, m] += ratio * b[j]
                d_k += ratio * c[j]
            d[k] = d_k

        return s_init, S_u, d

    def solve(
        self,
        current_temp: float,
        target_temp: float,
        previous_offset: float,
        outside_temps: np.ndarray,
        window_states: np.ndarray,
    ) -> MPCResult:
        """Löst das MPC-Problem."""
        cfg = self.config
        N = cfg.horizon_steps
        M = cfg.control_horizon

        # Pad/trim inputs
        if len(outside_temps) < N:
            outside_temps = np.pad(
                outside_temps, (0, N - len(outside_temps)), mode="edge"
            )
        else:
            outside_temps = outside_temps[:N]

        if len(window_states) < N:
            window_states = np.pad(
                window_states, (0, N - len(window_states)), mode="edge"
            )
        else:
            window_states = window_states[:N]

        start_time = time.time()

        try:
            # Build dynamics: T_pred = s_init * T_init + S_u @ u + d
            s_init, S_u, d = self._build_dynamics_matrices(
                outside_temps, window_states
            )

            # Constant part of predicted temperatures
            T_const = s_init * current_temp + d  # (N,)
            T_error_const = T_const - target_temp  # (N,)

            # Differencing matrix for smoothness: Δu = D @ u + d_vec
            # Δu[0] = u[0] - u_prev,  Δu[i] = u[i] - u[i-1]
            D = np.zeros((M, M))
            D[0, 0] = 1.0
            for i in range(1, M):
                D[i, i] = 1.0
                D[i, i - 1] = -1.0
            d_vec = np.zeros(M)
            d_vec[0] = -previous_offset

            # QP: min 0.5 * u^T H u + f^T u   s.t.  lb <= u <= ub
            # H = 2 * (w_c * S_u^T S_u + w_e * I + w_s * D^T D)
            # f = 2 * (w_c * S_u^T T_error_const + w_s * D^T d_vec)
            H = (
                cfg.weight_comfort * (S_u.T @ S_u)
                + cfg.weight_energy * np.eye(M)
                + cfg.weight_smoothness * (D.T @ D)
            )
            f = (
                cfg.weight_comfort * (S_u.T @ T_error_const)
                + cfg.weight_smoothness * (D.T @ d_vec)
            )

            # Solve box-constrained QP
            u_star = self._solve_box_qp(H, f, cfg.offset_min, cfg.offset_max)

            # Predicted temperatures
            predicted_temps = T_const + S_u @ u_star

            # Cost value
            T_err = predicted_temps - target_temp
            delta_u = D @ u_star + d_vec
            cost_value = float(
                cfg.weight_comfort * np.dot(T_err, T_err)
                + cfg.weight_energy * np.dot(u_star, u_star)
                + cfg.weight_smoothness * np.dot(delta_u, delta_u)
            )

            optimal_offset = float(u_star[0])
            status = "optimal"

        except Exception as e:
            logger.error("MPC solve failed: %s", e, exc_info=True)
            # Fallback: proportional control
            error = target_temp - current_temp
            optimal_offset = float(
                np.clip(-error * 2, cfg.offset_min, cfg.offset_max)
            )
            u_star = np.full(M, optimal_offset)
            predicted_temps = self.model.predict_horizon(
                current_temp,
                u_star,
                outside_temps,
                window_states,
                cfg.dt_minutes,
            )
            cost_value = float("inf")
            status = "error"

        solve_time_ms = (time.time() - start_time) * 1000
        self.solve_count += 1
        self.total_solve_time += solve_time_ms

        result = MPCResult(
            optimal_offset=optimal_offset,
            optimal_offsets=np.array(u_star),
            predicted_temps=np.array(predicted_temps),
            cost_value=cost_value,
            solve_time_ms=solve_time_ms,
            status=status,
        )

        if cfg.verbose:
            logger.debug(
                "MPC: offset=%.2f°C, cost=%.2f, time=%.1fms, status=%s",
                result.optimal_offset,
                result.cost_value,
                solve_time_ms,
                status,
            )
        return result

    @staticmethod
    def _solve_box_qp(
        H: np.ndarray,
        f: np.ndarray,
        lb: float,
        ub: float,
        max_iter: int = 200,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        Solve  min  0.5 * x^T H x + f^T x   s.t.  lb <= x <= ub

        Uses projected gradient descent with exact step size for quadratics.
        """
        M = len(f)

        # Start with unconstrained optimum, then clip
        try:
            u = np.linalg.solve(H, -f)
        except np.linalg.LinAlgError:
            u = np.zeros(M)
        u = np.clip(u, lb, ub)

        for _ in range(max_iter):
            grad = H @ u + f

            # Exact step size: α = grad^T grad / (grad^T H grad)
            Hg = H @ grad
            denom = grad @ Hg
            if denom <= 0:
                break
            alpha = (grad @ grad) / denom

            u_new = np.clip(u - alpha * grad, lb, ub)
            if np.max(np.abs(u_new - u)) < tol:
                break
            u = u_new

        return u

    def update_model(self, model: ThermalModel):
        """Aktualisiert das verwendete Modell."""
        self.model = model

    def update_config(self, **kwargs):
        """Aktualisiert Konfiguration und baut Problem neu."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def get_stats(self) -> Dict:
        """Gibt Statistiken zurück."""
        return {
            "solve_count": self.solve_count,
            "total_solve_time_ms": self.total_solve_time,
            "avg_solve_time_ms": self.total_solve_time / max(self.solve_count, 1),
        }
