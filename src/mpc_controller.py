"""
Model Predictive Controller (MPC)
=================================
Optimiert Temperatur-Offset über einen Prädiktionshorizont.
"""

import numpy as np
import cvxpy as cp
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from .model import ThermalModel, ThermalModelParams

logger = logging.getLogger(__name__)


@dataclass
class MPCConfig:
    """MPC Konfiguration."""
    
    # Horizonte
    horizon_steps: int = 20          # Anzahl Prädiktionsschritte
    control_horizon: int = 10        # Kontrollhorizont (Rest wird gehalten)
    dt_minutes: float = 5.0          # Zeitschritt [Minuten]
    
    # Gewichte
    weight_comfort: float = 1.0      # Abweichung von Solltemperatur
    weight_energy: float = 0.1       # Heizenergie (Offset)
    weight_smoothness: float = 0.05  # Stellgrößenänderungen
    
    # Constraints
    offset_min: float = -5.0         # Minimaler Offset [°C]
    offset_max: float = 5.0          # Maximaler Offset [°C]
    temp_min: float = 18.0           # Minimale Raumtemperatur
    temp_max: float = 26.0           # Maximale Raumtemperatur
    
    # Solver
    solver: str = "OSQP"             # OSQP, ECOS, SCS
    verbose: bool = False


@dataclass
class MPCResult:
    """Ergebnis einer MPC-Optimierung."""
    
    optimal_offset: float            # Anzuwendender Offset (erster Schritt)
    optimal_offsets: np.ndarray      # Alle optimalen Offsets
    predicted_temps: np.ndarray      # Vorhergesagte Temperaturen
    cost_value: float                # Kostenfunktionswert
    solve_time_ms: float             # Rechenzeit [ms]
    status: str                      # Solver-Status
    
    def to_dict(self) -> Dict:
        return {
            'optimal_offset': self.optimal_offset,
            'optimal_offsets': self.optimal_offsets.tolist(),
            'predicted_temps': self.predicted_temps.tolist(),
            'cost_value': self.cost_value,
            'solve_time_ms': self.solve_time_ms,
            'status': self.status,
        }


class MPCController:
    """
    Model Predictive Controller für Heizungsregelung.
    
    Optimiert den Temperatur-Offset um:
    1. Raumtemperatur nahe Sollwert zu halten (Komfort)
    2. Heizenergie zu minimieren
    3. Glatte Stellgrößenverläufe (weniger Verschleiß)
    
    Kostenfunktion:
    J = Σ [ w_c * (T - T_target)² + w_e * offset² + w_s * Δoffset² ]
    
    Subject to:
    - T(k+1) = f(T(k), offset(k), ...) (Systemdynamik)
    - offset_min ≤ offset ≤ offset_max
    - temp_min ≤ T ≤ temp_max
    """
    
    def __init__(
        self,
        model: ThermalModel,
        config: Optional[MPCConfig] = None,
    ):
        """
        Args:
            model: Thermisches Raummodell für Prädiktion
            config: MPC-Konfiguration
        """
        self.model = model
        self.config = config or MPCConfig()
        
        # Cache für Problem (wird bei erstem Solve erstellt)
        self._problem: Optional[cp.Problem] = None
        self._parameters: Dict[str, cp.Parameter] = {}
        self._variables: Dict[str, cp.Variable] = {}
        
        # Statistiken
        self.solve_count = 0
        self.total_solve_time = 0.0
        
        self._build_problem()
    
    def _build_problem(self):
        """
        Erstellt das CVXPY Optimierungsproblem.
        
        Für schnelle wiederholte Lösung mit Parametern.
        """
        cfg = self.config
        N = cfg.horizon_steps
        M = cfg.control_horizon
        
        # Variablen
        u = cp.Variable(M, name="offsets")        # Stellgrößen (Offsets)
        T = cp.Variable(N + 1, name="temps")      # Temperaturen
        
        # Parameter (werden bei jedem Solve aktualisiert)
        T_init = cp.Parameter(name="T_init")          # Starttemperatur
        T_target = cp.Parameter(name="T_target")      # Solltemperatur
        u_prev = cp.Parameter(name="u_prev")          # Vorheriger Offset
        
        # Außentemperatur-Profil (kann zeitvariant sein)
        T_out = cp.Parameter(N, name="T_out")
        
        # Fensterzustand-Profil
        window = cp.Parameter(N, name="window")
        
        # Modellparameter (werden aus self.model geholt)
        tau = cp.Parameter(name="tau", pos=True)
        k_heater = cp.Parameter(name="k_heater")
        k_outside = cp.Parameter(name="k_outside")
        k_window = cp.Parameter(name="k_window")
        
        # Parameter speichern
        self._parameters = {
            'T_init': T_init,
            'T_target': T_target,
            'u_prev': u_prev,
            'T_out': T_out,
            'window': window,
            'tau': tau,
            'k_heater': k_heater,
            'k_outside': k_outside,
            'k_window': k_window,
        }
        
        self._variables = {
            'offsets': u,
            'temps': T,
        }
        
        # Constraints
        constraints = []
        
        # Anfangsbedingung
        constraints.append(T[0] == T_init)
        
        # Systemdynamik (linearisiert)
        # T(k+1) = T(k) + dt/τ * (K_h * (-offset) + K_o * (T_out - T) + K_w * window * (T_out - T))
        dt = cfg.dt_minutes
        
        for k in range(N):
            # Offset: Verwende u[k] für k < M, sonst u[M-1] (hold)
            offset_k = u[k] if k < M else u[M - 1]
            
            delta_T_out = T_out[k] - T[k]
            
            # Vereinfachte lineare Dynamik
            # Für CVXPY muss das linear in den Variablen sein
            dT = (dt / tau) * (
                -k_heater * offset_k +
                k_outside * delta_T_out +
                k_window * window[k] * delta_T_out
            )
            
            # Da T[k] in delta_T_out vorkommt, müssen wir umformen:
            # dT = dt/τ * (-K_h * u + K_o * T_out - K_o * T + K_w * w * T_out - K_w * w * T)
            # T(k+1) = T(k) + dT
            # T(k+1) = T(k) * (1 - dt/τ * (K_o + K_w*w)) + dt/τ * (-K_h * u + (K_o + K_w*w) * T_out)
            
            coupling = k_outside + k_window * window[k]
            
            constraints.append(
                T[k + 1] == T[k] * (1 - dt / tau * coupling) +
                (dt / tau) * (-k_heater * offset_k + coupling * T_out[k])
            )
        
        # Offset-Grenzen
        constraints.append(u >= cfg.offset_min)
        constraints.append(u <= cfg.offset_max)
        
        # Temperatur-Grenzen (Soft-Constraint wäre besser für Feasibility)
        constraints.append(T >= cfg.temp_min)
        constraints.append(T <= cfg.temp_max)
        
        # Kostenfunktion
        cost = 0
        
        # Komfort: (T - T_target)²
        cost += cfg.weight_comfort * cp.sum_squares(T[1:] - T_target)
        
        # Energie: offset² (negativer Offset = heizen = Energie)
        # Wir minimieren |offset|, nicht offset²
        # Für quadratische: cost += cfg.weight_energy * cp.sum_squares(u)
        # Für absolute: cost += cfg.weight_energy * cp.norm(u, 1)
        cost += cfg.weight_energy * cp.sum_squares(u)
        
        # Glätte: (u[k] - u[k-1])²
        u_extended = cp.hstack([u_prev, u])
        cost += cfg.weight_smoothness * cp.sum_squares(cp.diff(u_extended))
        
        # Problem erstellen
        self._problem = cp.Problem(cp.Minimize(cost), constraints)
        
        logger.info(f"MPC problem built: N={N}, M={M}, {len(constraints)} constraints")
    
    def solve(
        self,
        current_temp: float,
        target_temp: float,
        previous_offset: float,
        outside_temps: np.ndarray,
        window_states: np.ndarray,
    ) -> MPCResult:
        """
        Löst das MPC-Problem.
        
        Args:
            current_temp: Aktuelle Raumtemperatur [°C]
            target_temp: Solltemperatur [°C]
            previous_offset: Zuletzt angewendeter Offset [°C]
            outside_temps: Außentemperatur-Vorhersage [N Werte]
            window_states: Fensterzustände [N Werte, 0 oder 1]
            
        Returns:
            MPCResult mit optimalem Offset und Diagnose
        """
        cfg = self.config
        N = cfg.horizon_steps
        
        # Arrays auf richtige Länge bringen
        if len(outside_temps) < N:
            outside_temps = np.pad(outside_temps, (0, N - len(outside_temps)), mode='edge')
        else:
            outside_temps = outside_temps[:N]
        
        if len(window_states) < N:
            window_states = np.pad(window_states, (0, N - len(window_states)), mode='edge')
        else:
            window_states = window_states[:N]
        
        # Parameter setzen
        self._parameters['T_init'].value = current_temp
        self._parameters['T_target'].value = target_temp
        self._parameters['u_prev'].value = previous_offset
        self._parameters['T_out'].value = outside_temps.astype(float)
        self._parameters['window'].value = window_states.astype(float)
        
        # Modellparameter
        p = self.model.params
        self._parameters['tau'].value = max(p.tau, 10.0)
        self._parameters['k_heater'].value = p.k_heater
        self._parameters['k_outside'].value = p.k_outside
        self._parameters['k_window'].value = p.k_window
        
        # Solve
        start_time = time.time()
        
        try:
            self._problem.solve(
                solver=getattr(cp, cfg.solver),
                verbose=cfg.verbose,
                warm_start=True,
            )
            status = self._problem.status
        except Exception as e:
            logger.error(f"MPC solve failed: {e}")
            status = "error"
        
        solve_time_ms = (time.time() - start_time) * 1000
        self.solve_count += 1
        self.total_solve_time += solve_time_ms
        
        # Ergebnisse extrahieren
        if status == cp.OPTIMAL or status == cp.OPTIMAL_INACCURATE:
            optimal_offsets = self._variables['offsets'].value
            predicted_temps = self._variables['temps'].value[1:]
            cost_value = self._problem.value
            optimal_offset = optimal_offsets[0]
        else:
            logger.warning(f"MPC solve status: {status}, using fallback")
            # Fallback: Proportionalregler
            error = target_temp - current_temp
            optimal_offset = np.clip(-error * 2, cfg.offset_min, cfg.offset_max)
            optimal_offsets = np.full(cfg.control_horizon, optimal_offset)
            predicted_temps = self.model.predict_horizon(
                current_temp, optimal_offsets,
                outside_temps, window_states, cfg.dt_minutes
            )
            cost_value = float('inf')
        
        result = MPCResult(
            optimal_offset=float(optimal_offset),
            optimal_offsets=np.array(optimal_offsets),
            predicted_temps=np.array(predicted_temps),
            cost_value=float(cost_value) if cost_value is not None else float('inf'),
            solve_time_ms=solve_time_ms,
            status=status,
        )
        
        logger.debug(
            f"MPC: offset={result.optimal_offset:.2f}°C, "
            f"cost={result.cost_value:.2f}, "
            f"time={solve_time_ms:.1f}ms"
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
            'solve_count': self.solve_count,
            'total_solve_time_ms': self.total_solve_time,
            'avg_solve_time_ms': self.total_solve_time / max(self.solve_count, 1),
        }


class SimpleMPCController:
    """
    Vereinfachter MPC ohne CVXPY (rein numpy-basiert).
    
    Verwendet Brute-Force Optimierung über diskrete Offset-Werte.
    Schneller für kleine Probleme, aber weniger elegant.
    """
    
    def __init__(
        self,
        model: ThermalModel,
        horizon_steps: int = 10,
        dt_minutes: float = 5.0,
        offset_values: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.horizon_steps = horizon_steps
        self.dt_minutes = dt_minutes
        
        # Diskrete Offset-Werte zum Testen
        self.offset_values = offset_values if offset_values is not None else \
            np.linspace(-5, 5, 21)  # -5 bis +5 in 0.5°C Schritten
    
    def solve(
        self,
        current_temp: float,
        target_temp: float,
        outside_temp: float,
        window_open: bool,
    ) -> float:
        """
        Findet optimalen Offset durch Enumeration.
        
        Vereinfachung: Konstanter Offset über Horizont.
        
        Returns:
            Optimaler Offset
        """
        best_offset = 0.0
        best_cost = float('inf')
        
        for offset in self.offset_values:
            # Simuliere Temperaturverlauf
            temps = self.model.predict_horizon(
                current_temp,
                np.full(self.horizon_steps, offset),
                np.full(self.horizon_steps, outside_temp),
                np.full(self.horizon_steps, 1 if window_open else 0),
                self.dt_minutes,
            )
            
            # Kosten: Abweichung von Sollwert + Energie
            cost = np.sum((temps - target_temp) ** 2) + 0.1 * offset ** 2
            
            if cost < best_cost:
                best_cost = cost
                best_offset = offset
        
        return best_offset


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test
    model = ThermalModel(
        initial_params=ThermalModelParams(
            tau=120,
            k_heater=0.5,
            k_outside=0.1,
            k_window=0.1,
        )
    )
    
    config = MPCConfig(
        horizon_steps=20,
        control_horizon=10,
        dt_minutes=5.0,
        weight_comfort=1.0,
        weight_energy=0.1,
        weight_smoothness=0.05,
    )
    
    mpc = MPCController(model, config)
    
    # Test solve
    result = mpc.solve(
        current_temp=19.0,
        target_temp=21.0,
        previous_offset=0.0,
        outside_temps=np.full(20, 5.0),
        window_states=np.zeros(20),
    )
    
    print(f"Optimal offset: {result.optimal_offset:.2f}°C")
    print(f"Predicted temps: {result.predicted_temps}")
    print(f"Solve time: {result.solve_time_ms:.1f}ms")
    print(f"Status: {result.status}")
