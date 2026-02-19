"""
Thermisches Raummodell
======================
Adaptives Modell mit Online-Parameteranpassung (RLS).
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ThermalModelParams:
    """Parameter des thermischen Modells."""

    tau: float = 120.0
    k_heater: float = 0.5
    k_outside: float = 0.1
    k_window: float = 0.05
    bias: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "tau": self.tau,
            "k_heater": self.k_heater,
            "k_outside": self.k_outside,
            "k_window": self.k_window,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ThermalModelParams":
        return cls(**d)


@dataclass
class RLSState:
    """Zustand des Recursive Least Squares Schätzers."""

    theta: np.ndarray = field(
        default_factory=lambda: np.array([1 / 120, 0.5, 0.1, 0.05, 0.0])
    )
    P: np.ndarray = field(default_factory=lambda: np.eye(5) * 1000)
    lambda_: float = 0.98
    n_updates: int = 0


class ThermalModel:
    """
    Adaptives thermisches Raummodell.

    Modellgleichung (diskret):
    T(k+1) = T(k) + dt/τ * (K_h * offset + K_o * (T_out - T) + K_w * window * (T_out - T) + bias)
    """

    def __init__(
        self,
        initial_params: Optional[ThermalModelParams] = None,
        forgetting_factor: float = 0.98,
    ):
        self.params = initial_params or ThermalModelParams()
        self.rls = RLSState(
            theta=self._params_to_theta(),
            lambda_=forgetting_factor,
        )
        self.param_history: List[Tuple[datetime, ThermalModelParams]] = []
        self.prediction_errors: List[float] = []

    def _params_to_theta(self) -> np.ndarray:
        p = self.params
        tau_inv = 1.0 / max(p.tau, 1.0)
        return np.array([
            tau_inv,
            p.k_heater * tau_inv,
            p.k_outside * tau_inv,
            p.k_window * tau_inv,
            p.bias * tau_inv,
        ])

    def _theta_to_params(self, theta: np.ndarray) -> ThermalModelParams:
        tau_inv = max(theta[0], 1e-6)
        tau = 1.0 / tau_inv
        return ThermalModelParams(
            tau=np.clip(tau, 10, 600),
            k_heater=np.clip(theta[1] / tau_inv, 0.01, 5.0),
            k_outside=np.clip(theta[2] / tau_inv, 0.0, 1.0),
            k_window=np.clip(theta[3] / tau_inv, 0.0, 1.0),
            bias=theta[4] / tau_inv,
        )

    def predict(
        self,
        current_temp: float,
        offset: float,
        outside_temp: float,
        window_open: bool,
        dt_minutes: float,
    ) -> float:
        """Sagt nächste Temperatur voraus."""
        p = self.params
        delta_t_outside = outside_temp - current_temp
        heating_effect = -p.k_heater * offset
        outside_effect = p.k_outside * delta_t_outside
        window_effect = p.k_window * delta_t_outside if window_open else 0
        dT_dt = (heating_effect + outside_effect + window_effect + p.bias) / p.tau
        return current_temp + dT_dt * dt_minutes

    def predict_horizon(
        self,
        current_temp: float,
        offsets: np.ndarray,
        outside_temps: np.ndarray,
        window_states: np.ndarray,
        dt_minutes: float,
    ) -> np.ndarray:
        """Sagt Temperaturen über einen Horizont voraus."""
        n = len(offsets)
        temps = np.zeros(n + 1)
        temps[0] = current_temp
        for i in range(n):
            temps[i + 1] = self.predict(
                temps[i],
                offsets[i],
                outside_temps[i] if i < len(outside_temps) else outside_temps[-1],
                bool(window_states[i]) if i < len(window_states) else False,
                dt_minutes,
            )
        return temps[1:]

    def update(
        self,
        prev_temp: float,
        current_temp: float,
        offset: float,
        outside_temp: float,
        window_open: bool,
        dt_minutes: float,
    ) -> float:
        """Aktualisiert Modellparameter mit neuer Messung (RLS)."""
        if dt_minutes <= 0:
            return 0.0

        dT_observed = current_temp - prev_temp
        delta_t_out = outside_temp - prev_temp
        window_flag = 1.0 if window_open else 0.0

        phi = np.array([
            dt_minutes * (20.0 - prev_temp),
            -dt_minutes * offset,
            dt_minutes * delta_t_out,
            dt_minutes * window_flag * delta_t_out,
            dt_minutes,
        ])

        dT_predicted = np.dot(phi, self.rls.theta)
        error = dT_observed - dT_predicted
        self.prediction_errors.append(error)

        P = self.rls.P
        lambda_ = self.rls.lambda_
        denominator = lambda_ + np.dot(phi, np.dot(P, phi))
        K = np.dot(P, phi) / denominator
        self.rls.theta = self.rls.theta + K * error
        self.rls.P = (P - np.outer(K, np.dot(phi, P))) / lambda_
        self.rls.P = (self.rls.P + self.rls.P.T) / 2

        self.params = self._theta_to_params(self.rls.theta)
        self.rls.n_updates += 1

        if self.rls.n_updates % 10 == 0:
            self.param_history.append(
                (datetime.now(), ThermalModelParams(**self.params.to_dict()))
            )
            logger.debug(
                "Model params updated: τ=%.1fmin, K_h=%.3f",
                self.params.tau,
                self.params.k_heater,
            )

        return error

    def get_confidence(self) -> Dict[str, float]:
        """Gibt Konfidenz für jeden Parameter zurück."""
        variances = np.diag(self.rls.P)
        confidences = 1.0 / (1.0 + np.sqrt(np.abs(variances)))
        return {
            "tau": confidences[0],
            "k_heater": confidences[1],
            "k_outside": confidences[2],
            "k_window": confidences[3],
            "bias": confidences[4],
        }

    def get_rmse(self, last_n: int = 100) -> float:
        """Root Mean Square Error der letzten n Vorhersagen."""
        if not self.prediction_errors:
            return float("inf")
        errors = self.prediction_errors[-last_n:]
        return float(np.sqrt(np.mean(np.array(errors) ** 2)))

    def reset_adaptation(self, keep_params: bool = True):
        """Setzt RLS-Zustand zurück."""
        if keep_params:
            self.rls.theta = self._params_to_theta()
        else:
            self.rls.theta = np.array([1 / 120, 0.5, 0.1, 0.05, 0.0])
        self.rls.P = np.eye(5) * 1000
        self.rls.n_updates = 0
        self.prediction_errors.clear()
        logger.info("Model adaptation reset")

    def save(self, filepath: str):
        """Speichert Modell in Datei."""
        from pathlib import Path

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "params": self.params.to_dict(),
            "rls_theta": self.rls.theta.tolist(),
            "rls_P": self.rls.P.tolist(),
            "rls_lambda": self.rls.lambda_,
            "rls_n_updates": self.rls.n_updates,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Model saved to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "ThermalModel":
        """Lädt Modell aus Datei."""
        with open(filepath) as f:
            data = json.load(f)
        model = cls(initial_params=ThermalModelParams.from_dict(data["params"]))
        model.rls.theta = np.array(data["rls_theta"])
        model.rls.P = np.array(data["rls_P"])
        model.rls.lambda_ = data["rls_lambda"]
        model.rls.n_updates = data["rls_n_updates"]
        logger.info("Model loaded from %s", filepath)
        return model

    def __repr__(self) -> str:
        p = self.params
        return (
            f"ThermalModel(τ={p.tau:.1f}min, K_h={p.k_heater:.3f}, "
            f"K_o={p.k_outside:.3f}, K_w={p.k_window:.3f}, "
            f"RMSE={self.get_rmse():.4f}°C, n={self.rls.n_updates})"
        )


def identify_from_step_response(
    times: np.ndarray,
    temps: np.ndarray,
    offset_step: float,
    outside_temp: float,
) -> ThermalModelParams:
    """Identifiziert Modellparameter aus Sprungantwort."""
    from scipy.optimize import curve_fit

    T_start = temps[0]
    T_end = temps[-1]

    def pt1_response(t, tau, T_final):
        return T_final + (T_start - T_final) * np.exp(-t / tau)

    try:
        popt, _ = curve_fit(
            pt1_response,
            times,
            temps,
            p0=[120, T_end],
            bounds=([10, temps.min() - 5], [600, temps.max() + 5]),
        )
        tau = popt[0]
        T_final = popt[1]
        delta_T = T_final - T_start
        k_heater = abs(delta_T / offset_step) if offset_step != 0 else 0.5
        return ThermalModelParams(
            tau=tau,
            k_heater=k_heater,
            k_outside=0.1,
            k_window=0.05,
        )
    except Exception as e:
        logger.error("Step response identification failed: %s", e)
        return ThermalModelParams()
