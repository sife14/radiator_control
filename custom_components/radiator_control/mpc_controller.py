"""
Model Predictive Controller (MPC)
=================================
Optimiert Temperatur-Offset über einen Prädiktionshorizont.
"""

import numpy as np
import cvxpy as cp
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass

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
    solver: str = "OSQP"
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
    """

    def __init__(
        self,
        model: ThermalModel,
        config: Optional[MPCConfig] = None,
    ):
        self.model = model
        self.config = config or MPCConfig()
        self._problem: Optional[cp.Problem] = None
        self._parameters: Dict[str, cp.Parameter] = {}
        self._variables: Dict[str, cp.Variable] = {}
        self.solve_count = 0
        self.total_solve_time = 0.0
        self._build_problem()

    def _build_problem(self):
        """Erstellt das CVXPY Optimierungsproblem."""
        cfg = self.config
        N = cfg.horizon_steps
        M = cfg.control_horizon

        u = cp.Variable(M, name="offsets")
        T = cp.Variable(N + 1, name="temps")

        T_init = cp.Parameter(name="T_init")
        T_target = cp.Parameter(name="T_target")
        u_prev = cp.Parameter(name="u_prev")
        T_out = cp.Parameter(N, name="T_out")
        window = cp.Parameter(N, name="window")
        tau = cp.Parameter(name="tau", pos=True)
        k_heater = cp.Parameter(name="k_heater")
        k_outside = cp.Parameter(name="k_outside")
        k_window = cp.Parameter(name="k_window")

        self._parameters = {
            "T_init": T_init,
            "T_target": T_target,
            "u_prev": u_prev,
            "T_out": T_out,
            "window": window,
            "tau": tau,
            "k_heater": k_heater,
            "k_outside": k_outside,
            "k_window": k_window,
        }
        self._variables = {"offsets": u, "temps": T}

        constraints = [T[0] == T_init]
        dt = cfg.dt_minutes

        for k in range(N):
            offset_k = u[k] if k < M else u[M - 1]
            coupling = k_outside + k_window * window[k]
            constraints.append(
                T[k + 1]
                == T[k] * (1 - dt / tau * coupling)
                + (dt / tau) * (-k_heater * offset_k + coupling * T_out[k])
            )

        constraints.append(u >= cfg.offset_min)
        constraints.append(u <= cfg.offset_max)
        constraints.append(T >= cfg.temp_min)
        constraints.append(T <= cfg.temp_max)

        cost = 0
        cost += cfg.weight_comfort * cp.sum_squares(T[1:] - T_target)
        cost += cfg.weight_energy * cp.sum_squares(u)
        u_extended = cp.hstack([u_prev, u])
        cost += cfg.weight_smoothness * cp.sum_squares(cp.diff(u_extended))

        self._problem = cp.Problem(cp.Minimize(cost), constraints)
        logger.info(
            "MPC problem built: N=%d, M=%d, %d constraints",
            N, M, len(constraints),
        )

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

        if len(outside_temps) < N:
            outside_temps = np.pad(outside_temps, (0, N - len(outside_temps)), mode="edge")
        else:
            outside_temps = outside_temps[:N]

        if len(window_states) < N:
            window_states = np.pad(window_states, (0, N - len(window_states)), mode="edge")
        else:
            window_states = window_states[:N]

        self._parameters["T_init"].value = current_temp
        self._parameters["T_target"].value = target_temp
        self._parameters["u_prev"].value = previous_offset
        self._parameters["T_out"].value = outside_temps.astype(float)
        self._parameters["window"].value = window_states.astype(float)

        p = self.model.params
        self._parameters["tau"].value = max(p.tau, 10.0)
        self._parameters["k_heater"].value = p.k_heater
        self._parameters["k_outside"].value = p.k_outside
        self._parameters["k_window"].value = p.k_window

        start_time = time.time()

        try:
            self._problem.solve(
                solver=getattr(cp, cfg.solver),
                verbose=cfg.verbose,
                warm_start=True,
            )
            status = self._problem.status
        except Exception as e:
            logger.error("MPC solve failed: %s", e)
            status = "error"

        solve_time_ms = (time.time() - start_time) * 1000
        self.solve_count += 1
        self.total_solve_time += solve_time_ms

        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            optimal_offsets = self._variables["offsets"].value
            predicted_temps = self._variables["temps"].value[1:]
            cost_value = self._problem.value
            optimal_offset = optimal_offsets[0]
        else:
            logger.warning("MPC solve status: %s, using fallback", status)
            error = target_temp - current_temp
            optimal_offset = np.clip(-error * 2, cfg.offset_min, cfg.offset_max)
            optimal_offsets = np.full(cfg.control_horizon, optimal_offset)
            predicted_temps = self.model.predict_horizon(
                current_temp,
                optimal_offsets,
                outside_temps,
                window_states,
                cfg.dt_minutes,
            )
            cost_value = float("inf")

        result = MPCResult(
            optimal_offset=float(optimal_offset),
            optimal_offsets=np.array(optimal_offsets),
            predicted_temps=np.array(predicted_temps),
            cost_value=float(cost_value) if cost_value is not None else float("inf"),
            solve_time_ms=solve_time_ms,
            status=status,
        )

        logger.debug(
            "MPC: offset=%.2f°C, cost=%.2f, time=%.1fms",
            result.optimal_offset,
            result.cost_value,
            solve_time_ms,
        )
        return result

    def update_model(self, model: ThermalModel):
        """Aktualisiert das verwendete Modell."""
        self.model = model

    def update_config(self, **kwargs):
        """Aktualisiert Konfiguration und baut Problem neu."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._build_problem()

    def get_stats(self) -> Dict:
        """Gibt Statistiken zurück."""
        return {
            "solve_count": self.solve_count,
            "total_solve_time_ms": self.total_solve_time,
            "avg_solve_time_ms": self.total_solve_time / max(self.solve_count, 1),
        }
