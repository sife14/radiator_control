"""
DataUpdateCoordinator für Radiator Control
==========================================
Zentrale Regelschleife als HA DataUpdateCoordinator.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_TEMPERATURE,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    DOMAIN,
    CONF_THERMOSTAT_ENTITY,
    CONF_TEMP_SENSOR_ENTITY,
    CONF_WINDOW_SENSOR_ENTITY,
    CONF_OUTSIDE_TEMP_ENTITY,
    CONF_TEMP_CALIBRATION_ENTITY,
    CONF_MPC_HORIZON,
    CONF_MPC_CONTROL_HORIZON,
    CONF_MPC_WEIGHT_COMFORT,
    CONF_MPC_WEIGHT_ENERGY,
    CONF_MPC_WEIGHT_SMOOTHNESS,
    CONF_OFFSET_MIN,
    CONF_OFFSET_MAX,
    CONF_SAMPLE_TIME,
    CONF_WINDOW_ACTION,
    CONF_WINDOW_OFF_DELAY,
    CONF_MODEL_FORGETTING_FACTOR,
    CONF_MODEL_INITIAL_TAU,
    CONF_MODEL_INITIAL_K_HEATER,
    DEFAULT_MPC_HORIZON,
    DEFAULT_MPC_CONTROL_HORIZON,
    DEFAULT_MPC_WEIGHT_COMFORT,
    DEFAULT_MPC_WEIGHT_ENERGY,
    DEFAULT_MPC_WEIGHT_SMOOTHNESS,
    DEFAULT_OFFSET_MIN,
    DEFAULT_OFFSET_MAX,
    DEFAULT_SAMPLE_TIME,
    DEFAULT_WINDOW_ACTION,
    DEFAULT_WINDOW_OFF_DELAY,
    DEFAULT_MODEL_FORGETTING_FACTOR,
    DEFAULT_MODEL_INITIAL_TAU,
    DEFAULT_MODEL_INITIAL_K_HEATER,
    WINDOW_ACTION_TURN_OFF,
)
from .database import Database, Measurement
from .model import ThermalModel, ThermalModelParams
from .mpc_controller import MPCController, MPCConfig

logger = logging.getLogger(__name__)


class RadiatorControlData:
    """Container for coordinator data exposed to entities."""

    def __init__(self):
        self.room_temp: Optional[float] = None
        self.target_temp: Optional[float] = None
        self.outside_temp: Optional[float] = None
        self.offset: float = 0.0
        self.window_open: bool = False
        self.hvac_mode: Optional[str] = None
        self.hvac_action: Optional[str] = None
        self.heating_active: bool = False
        self.mode: str = "idle"  # idle, control, window_open, experiment
        self.control_enabled: bool = False
        self.last_update: Optional[datetime] = None

        # Model info
        self.model_tau: float = 0.0
        self.model_k_heater: float = 0.0
        self.model_k_outside: float = 0.0
        self.model_rmse: float = 0.0
        self.model_updates: int = 0

        # MPC info
        self.mpc_cost: float = 0.0
        self.mpc_solve_time_ms: float = 0.0

        # Experiment
        self.experiment_type: Optional[str] = None
        self.experiment_progress: float = 0.0


class RadiatorControlCoordinator(DataUpdateCoordinator[RadiatorControlData]):
    """Coordinator for Radiator Control MPC."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        self.entry = entry
        self._data = RadiatorControlData()

        # Entity IDs from config
        self._thermostat_entity: str = entry.data[CONF_THERMOSTAT_ENTITY]
        self._temp_sensor_entity: Optional[str] = entry.data.get(CONF_TEMP_SENSOR_ENTITY)
        self._window_sensor_entity: Optional[str] = entry.data.get(CONF_WINDOW_SENSOR_ENTITY)
        self._outside_temp_entity: Optional[str] = entry.data.get(CONF_OUTSIDE_TEMP_ENTITY)
        self._calibration_entity: Optional[str] = entry.data.get(CONF_TEMP_CALIBRATION_ENTITY)

        # Storage path
        storage_dir = Path(hass.config.path("radiator_control"))
        storage_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = str(storage_dir / "model.json")
        self._db_path = str(storage_dir / "measurements.db")

        # Core components
        self.database = Database(self._db_path)
        self.model = self._load_or_create_model()
        self.mpc: Optional[MPCController] = None

        # Control state
        self._previous_temp: Optional[float] = None
        self._previous_time: Optional[datetime] = None
        self._previous_offset: float = 0.0
        self._previous_window_open: bool = False
        self._saved_hvac_mode: Optional[str] = None
        self._window_open_since: Optional[datetime] = None
        self._model_save_counter: int = 0

        # Experiment
        self._experiment_task: Optional[asyncio.Task] = None

        sample_time = self._get_option(CONF_SAMPLE_TIME, DEFAULT_SAMPLE_TIME)

        super().__init__(
            hass,
            logger,
            name=DOMAIN,
            update_interval=timedelta(seconds=sample_time),
        )

    def _get_option(self, key: str, default: Any) -> Any:
        """Get option from entry options or data."""
        return self.entry.options.get(key, self.entry.data.get(key, default))

    def _load_or_create_model(self) -> ThermalModel:
        """Load existing model or create a new one."""
        model_path = Path(self._model_path)
        if model_path.exists():
            try:
                model = ThermalModel.load(str(model_path))
                logger.info("Loaded existing thermal model")
                return model
            except Exception as e:
                logger.warning("Failed to load model, creating new: %s", e)

        tau = self._get_option(CONF_MODEL_INITIAL_TAU, DEFAULT_MODEL_INITIAL_TAU)
        k_heater = self._get_option(CONF_MODEL_INITIAL_K_HEATER, DEFAULT_MODEL_INITIAL_K_HEATER)
        ff = self._get_option(CONF_MODEL_FORGETTING_FACTOR, DEFAULT_MODEL_FORGETTING_FACTOR)

        return ThermalModel(
            initial_params=ThermalModelParams(tau=tau, k_heater=k_heater),
            forgetting_factor=ff,
        )

    def _build_mpc(self) -> MPCController:
        """Build MPC controller from current options."""
        horizon = int(self._get_option(CONF_MPC_HORIZON, DEFAULT_MPC_HORIZON))
        return MPCController(
            model=self.model,
            config=MPCConfig(
                horizon_steps=horizon // 5,
                control_horizon=int(self._get_option(
                    CONF_MPC_CONTROL_HORIZON, DEFAULT_MPC_CONTROL_HORIZON
                )),
                dt_minutes=5.0,
                weight_comfort=float(self._get_option(
                    CONF_MPC_WEIGHT_COMFORT, DEFAULT_MPC_WEIGHT_COMFORT
                )),
                weight_energy=float(self._get_option(
                    CONF_MPC_WEIGHT_ENERGY, DEFAULT_MPC_WEIGHT_ENERGY
                )),
                weight_smoothness=float(self._get_option(
                    CONF_MPC_WEIGHT_SMOOTHNESS, DEFAULT_MPC_WEIGHT_SMOOTHNESS
                )),
                offset_min=float(self._get_option(
                    CONF_OFFSET_MIN, DEFAULT_OFFSET_MIN
                )),
                offset_max=float(self._get_option(
                    CONF_OFFSET_MAX, DEFAULT_OFFSET_MAX
                )),
            ),
        )

    # -------------------------------------------------------------------------
    # State reading helpers
    # -------------------------------------------------------------------------

    def _get_float_state(self, entity_id: Optional[str]) -> Optional[float]:
        """Read a numeric state value."""
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN, None):
            return None
        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    def _get_thermostat_data(self) -> dict[str, Any]:
        """Read thermostat state and attributes."""
        state = self.hass.states.get(self._thermostat_entity)
        if state is None:
            return {}
        attrs = state.attributes or {}
        return {
            "hvac_mode": state.state,
            "current_temperature": attrs.get("current_temperature"),
            "temperature": attrs.get(ATTR_TEMPERATURE),
            "hvac_action": attrs.get("hvac_action"),
        }

    def _is_window_open(self) -> bool:
        """Check window sensor state."""
        if not self._window_sensor_entity:
            return False
        state = self.hass.states.get(self._window_sensor_entity)
        if state is None:
            return False
        return state.state == STATE_ON

    # -------------------------------------------------------------------------
    # HA service calls
    # -------------------------------------------------------------------------

    async def _set_temperature_offset(self, offset: float) -> None:
        """Set the temperature calibration offset."""
        calibration_entity = self._calibration_entity
        if not calibration_entity:
            # Try to derive from thermostat name
            thermostat_name = self._thermostat_entity.split(".")[-1]
            calibration_entity = f"number.{thermostat_name}_local_temperature_calibration"

        state = self.hass.states.get(calibration_entity)
        if state is not None and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            await self.hass.services.async_call(
                "number",
                "set_value",
                {"entity_id": calibration_entity, "value": offset},
                blocking=True,
            )
            logger.debug("Set temperature calibration to %.1f°C", offset)
        else:
            logger.warning(
                "Calibration entity %s not available, cannot set offset",
                calibration_entity,
            )

    async def _set_hvac_mode(self, mode: str) -> None:
        """Set HVAC mode on thermostat."""
        await self.hass.services.async_call(
            "climate",
            "set_hvac_mode",
            {"entity_id": self._thermostat_entity, "hvac_mode": mode},
            blocking=True,
        )
        logger.info("Set thermostat HVAC mode to %s", mode)

    async def _turn_off_thermostat(self) -> None:
        """Turn off the thermostat."""
        await self._set_hvac_mode("off")

    async def _turn_on_thermostat(self, mode: str = "heat") -> None:
        """Turn on the thermostat."""
        await self._set_hvac_mode(mode)

    # -------------------------------------------------------------------------
    # Control logic
    # -------------------------------------------------------------------------

    def set_control_enabled(self, enabled: bool) -> None:
        """Enable or disable active control."""
        self._data.control_enabled = enabled
        if not enabled:
            self._data.mode = "idle"
            # Reset offset asynchronously
            self.hass.async_create_task(self._set_temperature_offset(0))
        logger.info("Control %s", "enabled" if enabled else "disabled")

    async def _async_update_data(self) -> RadiatorControlData:
        """Fetch data and run control loop."""
        now = datetime.now()

        # Read sensors
        thermo_data = self._get_thermostat_data()

        # Room temperature: prefer separate sensor
        room_temp = self._get_float_state(self._temp_sensor_entity)
        if room_temp is None:
            room_temp = thermo_data.get("current_temperature")
        if room_temp is None:
            room_temp = 20.0

        target_temp = thermo_data.get("temperature", 21.0)
        outside_temp = self._get_float_state(self._outside_temp_entity)
        if outside_temp is None:
            outside_temp = 5.0

        window_open = self._is_window_open()
        hvac_mode = thermo_data.get("hvac_mode")
        hvac_action = thermo_data.get("hvac_action")
        heating_active = hvac_action == "heating"

        # Update data
        self._data.room_temp = room_temp
        self._data.target_temp = target_temp
        self._data.outside_temp = outside_temp
        self._data.window_open = window_open
        self._data.hvac_mode = hvac_mode
        self._data.hvac_action = hvac_action
        self._data.heating_active = heating_active
        self._data.last_update = now

        # Update model info
        self._data.model_tau = self.model.params.tau
        self._data.model_k_heater = self.model.params.k_heater
        self._data.model_k_outside = self.model.params.k_outside
        self._data.model_rmse = self.model.get_rmse()
        self._data.model_updates = self.model.rls.n_updates

        # If control not enabled, just collect data
        if not self._data.control_enabled:
            self._data.mode = "idle"
            return self._data

        # Experiment running?
        if self._experiment_task and not self._experiment_task.done():
            self._data.mode = "experiment"
            return self._data

        # Control logic
        window_action = self._get_option(CONF_WINDOW_ACTION, DEFAULT_WINDOW_ACTION)
        window_off_delay = int(self._get_option(CONF_WINDOW_OFF_DELAY, DEFAULT_WINDOW_OFF_DELAY))
        offset_max = float(self._get_option(CONF_OFFSET_MAX, DEFAULT_OFFSET_MAX))

        new_offset = 0.0

        if window_open:
            self._data.mode = "window_open"

            if window_action == WINDOW_ACTION_TURN_OFF:
                if self._window_open_since is None:
                    self._window_open_since = now

                elapsed = (now - self._window_open_since).total_seconds()
                if elapsed >= window_off_delay:
                    if hvac_mode and hvac_mode != "off":
                        if self._saved_hvac_mode is None:
                            self._saved_hvac_mode = hvac_mode
                        await self._turn_off_thermostat()
                        logger.info(
                            "Window open for %ds - turned thermostat OFF", elapsed
                        )
                    new_offset = 0
                else:
                    new_offset = offset_max
            else:
                new_offset = offset_max
        else:
            # Window closed
            self._window_open_since = None
            self._data.mode = "control"

            # Restore thermostat if it was turned off
            if self._saved_hvac_mode is not None and hvac_mode == "off":
                await self._turn_on_thermostat(self._saved_hvac_mode)
                logger.info(
                    "Window closed - restored thermostat to %s",
                    self._saved_hvac_mode,
                )
                self._saved_hvac_mode = None

            # Update model (only with closed window data)
            if (
                self._previous_temp is not None
                and self._previous_time is not None
            ):
                dt_minutes = (now - self._previous_time).total_seconds() / 60
                if dt_minutes > 0 and not self._previous_window_open:
                    self.model.update(
                        prev_temp=self._previous_temp,
                        current_temp=room_temp,
                        offset=self._previous_offset,
                        outside_temp=outside_temp,
                        window_open=False,
                        dt_minutes=dt_minutes,
                    )

            # MPC optimization
            if self.mpc is None:
                self.mpc = self._build_mpc()

            mpc_result = await self.hass.async_add_executor_job(
                self.mpc.solve,
                room_temp,
                target_temp,
                self._previous_offset,
                np.full(self.mpc.config.horizon_steps, outside_temp),
                np.zeros(self.mpc.config.horizon_steps),
            )

            new_offset = mpc_result.optimal_offset
            self._data.mpc_cost = mpc_result.cost_value
            self._data.mpc_solve_time_ms = mpc_result.solve_time_ms

            logger.info(
                "MPC: T=%.1f°C → target=%.1f°C, offset=%.2f°C, cost=%.2f",
                room_temp,
                target_temp,
                new_offset,
                mpc_result.cost_value,
            )

            # Log controller step
            self.database.log_controller_step(
                controller_type="mpc",
                predicted_temps=mpc_result.predicted_temps.tolist(),
                optimal_offsets=mpc_result.optimal_offsets.tolist(),
                cost_value=mpc_result.cost_value,
                solve_time_ms=mpc_result.solve_time_ms,
            )

        # Apply offset
        await self._set_temperature_offset(new_offset)
        self._data.offset = new_offset

        # Save measurement
        measurement = Measurement(
            timestamp=now,
            room_temp=room_temp,
            outside_temp=outside_temp,
            window_open=window_open,
            heating_active=heating_active,
            control_offset=new_offset,
            target_temp=target_temp,
            mode=self._data.mode,
        )
        await self.hass.async_add_executor_job(
            self.database.insert_measurement, measurement
        )

        # Log training sample
        reward = self._calculate_reward(
            room_temp, target_temp, new_offset, window_open
        )
        await self.hass.async_add_executor_job(
            self.database.log_training_sample,
            now,
            {
                "room_temp": room_temp,
                "outside_temp": outside_temp,
                "target_temp": target_temp,
                "window_open": window_open,
                "previous_temp": self._previous_temp,
                "previous_offset": self._previous_offset,
                "heating_active": heating_active,
            },
            {"offset": new_offset},
            reward,
            self.model.params.to_dict(),
        )

        # Update previous state
        self._previous_temp = room_temp
        self._previous_time = now
        self._previous_offset = new_offset
        self._previous_window_open = window_open

        # Periodically save model
        self._model_save_counter += 1
        if self._model_save_counter % 100 == 0:
            await self.hass.async_add_executor_job(
                self.model.save, self._model_path
            )

        return self._data

    def _calculate_reward(
        self,
        current_temp: float,
        target_temp: float,
        offset: float,
        window_open: bool,
    ) -> float:
        """Calculate reward for RL training."""
        if window_open:
            return 0.0
        temp_error = abs(current_temp - target_temp)
        comfort_reward = float(np.exp(-0.5 * (temp_error / 0.5) ** 2))
        energy_penalty = 0.1 * max(0, -offset)
        return comfort_reward - energy_penalty

    # -------------------------------------------------------------------------
    # Experiments
    # -------------------------------------------------------------------------

    async def start_experiment(self, experiment_type: str) -> None:
        """Start an experiment."""
        if self._experiment_task and not self._experiment_task.done():
            logger.warning("Experiment already running")
            return

        from .experiments import run_experiment

        self._data.experiment_type = experiment_type
        self._data.experiment_progress = 0.0
        self._data.mode = "experiment"

        self._experiment_task = self.hass.async_create_task(
            run_experiment(self, experiment_type)
        )

    async def stop_experiment(self) -> None:
        """Stop running experiment."""
        if self._experiment_task and not self._experiment_task.done():
            self._experiment_task.cancel()
        self._data.experiment_type = None
        self._data.experiment_progress = 0.0
        self._data.mode = "idle" if not self._data.control_enabled else "control"
        # Reset offset
        await self._set_temperature_offset(0)

    def reset_model(self) -> None:
        """Reset the thermal model to initial values."""
        tau = self._get_option(CONF_MODEL_INITIAL_TAU, DEFAULT_MODEL_INITIAL_TAU)
        k_heater = self._get_option(
            CONF_MODEL_INITIAL_K_HEATER, DEFAULT_MODEL_INITIAL_K_HEATER
        )
        ff = self._get_option(
            CONF_MODEL_FORGETTING_FACTOR, DEFAULT_MODEL_FORGETTING_FACTOR
        )
        self.model = ThermalModel(
            initial_params=ThermalModelParams(tau=tau, k_heater=k_heater),
            forgetting_factor=ff,
        )
        self.mpc = None  # Will be rebuilt on next update
        self.model.save(self._model_path)
        logger.info("Model reset to initial values")

    def rebuild_mpc(self) -> None:
        """Rebuild MPC with current options (called after options update)."""
        self.mpc = None  # Will be rebuilt on next update

    async def async_shutdown(self) -> None:
        """Clean shutdown."""
        if self._experiment_task and not self._experiment_task.done():
            self._experiment_task.cancel()

        # Reset offset
        try:
            await self._set_temperature_offset(0)
        except Exception:
            pass

        # Save model
        try:
            await self.hass.async_add_executor_job(
                self.model.save, self._model_path
            )
        except Exception:
            pass

        logger.info("Radiator Control coordinator shut down")
