"""
Experiment Runner für Home Assistant Integration
=================================================
Systemidentifikation via Step-Response, PRBS und Relay-Feedback.
Arbeitet direkt mit dem Coordinator statt mit REST-API.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

from .database import Measurement
from .model import ThermalModelParams, identify_from_step_response

if TYPE_CHECKING:
    from .coordinator import RadiatorControlCoordinator

logger = logging.getLogger(__name__)

# Default experiment parameters
STEP_OFFSET = -3.0
STEP_DURATION_MINUTES = 120
PRE_SETTLE_MINUTES = 15
PRBS_AMPLITUDE = 2.0
PRBS_MIN_DURATION = 15
PRBS_TOTAL_DURATION = 240
RELAY_HYSTERESIS = 0.3
RELAY_MAX_DURATION = 180
SAMPLE_INTERVAL = 60


async def run_experiment(
    coordinator: RadiatorControlCoordinator,
    experiment_type: str,
) -> None:
    """
    Run an experiment.

    This is the entry point called by the coordinator.
    """
    try:
        if experiment_type == "step":
            await _run_step_response(coordinator)
        elif experiment_type == "prbs":
            await _run_prbs(coordinator)
        elif experiment_type == "relay":
            await _run_relay_feedback(coordinator)
        else:
            logger.error("Unknown experiment type: %s", experiment_type)
    except asyncio.CancelledError:
        logger.info("Experiment %s cancelled", experiment_type)
    except Exception as e:
        logger.error("Experiment %s failed: %s", experiment_type, e, exc_info=True)
    finally:
        # Reset offset
        try:
            await coordinator._set_temperature_offset(0)
        except Exception:
            pass
        coordinator._data.experiment_type = None
        coordinator._data.experiment_progress = 0.0
        if coordinator._data.control_enabled:
            coordinator._data.mode = "control"
        else:
            coordinator._data.mode = "idle"
        coordinator.async_set_updated_data(coordinator._data)


async def _read_sensor(coordinator: RadiatorControlCoordinator) -> dict:
    """Read all sensor values from HA states."""
    thermo_data = coordinator._get_thermostat_data()

    room_temp = coordinator._get_float_state(coordinator._temp_sensor_entity)
    if room_temp is None:
        room_temp = thermo_data.get("current_temperature", 20.0)

    outside_temp = coordinator._get_float_state(coordinator._outside_temp_entity)
    window_open = coordinator._is_window_open()
    target_temp = thermo_data.get("temperature", 21.0)
    hvac_action = thermo_data.get("hvac_action")

    return {
        "room_temp": room_temp,
        "outside_temp": outside_temp or 5.0,
        "window_open": window_open,
        "target_temp": target_temp,
        "heating_active": hvac_action == "heating",
    }


async def _collect_and_store(
    coordinator: RadiatorControlCoordinator,
    offset: float,
    mode: str = "experiment",
) -> dict:
    """Set offset, read sensors, store measurement, and return data."""
    await coordinator._set_temperature_offset(offset)
    await asyncio.sleep(2)

    data = await _read_sensor(coordinator)

    measurement = Measurement(
        timestamp=datetime.now(),
        room_temp=data["room_temp"],
        outside_temp=data["outside_temp"],
        window_open=data["window_open"],
        heating_active=data["heating_active"],
        control_offset=offset,
        target_temp=data["target_temp"],
        mode=mode,
    )
    await coordinator.hass.async_add_executor_job(
        coordinator.database.insert_measurement, measurement
    )

    # Update coordinator data for entity display
    coordinator._data.room_temp = data["room_temp"]
    coordinator._data.outside_temp = data["outside_temp"]
    coordinator._data.window_open = data["window_open"]
    coordinator._data.offset = offset
    coordinator.async_set_updated_data(coordinator._data)

    return data


def _update_progress(
    coordinator: RadiatorControlCoordinator,
    progress: float,
) -> None:
    """Update experiment progress."""
    coordinator._data.experiment_progress = progress
    coordinator.async_set_updated_data(coordinator._data)


# =============================================================================
# Step Response
# =============================================================================


async def _run_step_response(
    coordinator: RadiatorControlCoordinator,
    initial_offset: float = 0.0,
    step_offset: float = STEP_OFFSET,
    duration_minutes: int = STEP_DURATION_MINUTES,
) -> None:
    """Run a step response experiment."""
    exp_id = await coordinator.hass.async_add_executor_job(
        coordinator.database.start_experiment,
        f"Step Response {datetime.now().strftime('%Y%m%d_%H%M')}",
        "step_response",
        {
            "initial_offset": initial_offset,
            "step_offset": step_offset,
            "duration_minutes": duration_minutes,
        },
    )

    times: list[float] = []
    temps: list[float] = []
    offsets: list[float] = []
    outside_temps: list[float] = []

    try:
        # Phase 1: Settle
        logger.info("Step response: settling phase (%d min)", PRE_SETTLE_MINUTES)
        settle_samples = PRE_SETTLE_MINUTES * 60 // SAMPLE_INTERVAL
        for i in range(settle_samples):
            data = await _collect_and_store(coordinator, initial_offset)
            progress = 0.1 * (i / settle_samples)
            _update_progress(coordinator, progress)
            await asyncio.sleep(SAMPLE_INTERVAL)

        # Phase 2: Step
        logger.info(
            "Step response: heating phase (%d min, offset=%.1f)",
            duration_minutes,
            step_offset,
        )
        step_start = datetime.now()
        total_samples = duration_minutes * 60 // SAMPLE_INTERVAL

        for i in range(total_samples):
            data = await _collect_and_store(coordinator, step_offset)

            if data["window_open"]:
                logger.warning("Window opened during step response - aborting")
                raise asyncio.CancelledError("Window opened")

            elapsed = (datetime.now() - step_start).total_seconds() / 60
            times.append(elapsed)
            temps.append(data["room_temp"])
            offsets.append(step_offset)
            outside_temps.append(data["outside_temp"])

            progress = 0.1 + 0.8 * (i / total_samples)
            _update_progress(coordinator, progress)
            await asyncio.sleep(SAMPLE_INTERVAL)

        # Phase 3: Analysis
        logger.info("Step response: analyzing data")
        _update_progress(coordinator, 0.95)

        times_arr = np.array(times)
        temps_arr = np.array(temps)

        identified_params = identify_from_step_response(
            times=times_arr,
            temps=temps_arr,
            offset_step=step_offset - initial_offset,
            outside_temp=np.mean(outside_temps),
        )

        # Update model
        coordinator.model.params = identified_params
        await coordinator.hass.async_add_executor_job(
            coordinator.model.save, coordinator._model_path
        )
        coordinator.mpc = None  # Rebuild on next update

        result = {
            "identified_params": identified_params.to_dict(),
            "metrics": {
                "T_initial": float(temps_arr[0]),
                "T_final": float(temps_arr[-1]),
                "T_change": float(temps_arr[-1] - temps_arr[0]),
                "n_samples": len(temps),
            },
        }

        await coordinator.hass.async_add_executor_job(
            coordinator.database.end_experiment, exp_id, result
        )
        _update_progress(coordinator, 1.0)
        logger.info(
            "Step response completed: τ=%.0f, K_h=%.3f",
            identified_params.tau,
            identified_params.k_heater,
        )

    except asyncio.CancelledError:
        await coordinator.hass.async_add_executor_job(
            coordinator.database.end_experiment, exp_id, {"status": "cancelled"}
        )
        raise


# =============================================================================
# PRBS
# =============================================================================


async def _run_prbs(
    coordinator: RadiatorControlCoordinator,
    amplitude: float = PRBS_AMPLITUDE,
    min_duration: int = PRBS_MIN_DURATION,
    total_duration: int = PRBS_TOTAL_DURATION,
) -> None:
    """Run a PRBS experiment."""
    exp_id = await coordinator.hass.async_add_executor_job(
        coordinator.database.start_experiment,
        f"PRBS {datetime.now().strftime('%Y%m%d_%H%M')}",
        "prbs",
        {
            "amplitude": amplitude,
            "min_duration": min_duration,
            "total_duration": total_duration,
        },
    )

    times: list[float] = []
    temps: list[float] = []
    offsets: list[float] = []
    outside_temps: list[float] = []

    try:
        logger.info("PRBS experiment started (%d min)", total_duration)
        elapsed_minutes = 0.0
        current_offset = -amplitude
        hold_time = min_duration + np.random.randint(0, min_duration)
        hold_counter = 0.0
        switches = 0

        while elapsed_minutes < total_duration:
            data = await _collect_and_store(coordinator, current_offset)

            if data["window_open"]:
                logger.warning("Window opened during PRBS - aborting")
                raise asyncio.CancelledError("Window opened")

            times.append(elapsed_minutes)
            temps.append(data["room_temp"])
            offsets.append(current_offset)
            outside_temps.append(data["outside_temp"])

            hold_counter += SAMPLE_INTERVAL / 60
            elapsed_minutes += SAMPLE_INTERVAL / 60

            if hold_counter >= hold_time:
                current_offset = -current_offset
                hold_time = min_duration + np.random.randint(0, min_duration)
                hold_counter = 0.0
                switches += 1

            progress = elapsed_minutes / total_duration
            _update_progress(coordinator, min(progress, 0.99))
            await asyncio.sleep(SAMPLE_INTERVAL)

        result = {
            "metrics": {
                "n_samples": len(times),
                "n_switches": switches,
                "temp_min": float(np.min(temps)),
                "temp_max": float(np.max(temps)),
                "temp_std": float(np.std(temps)),
            }
        }
        await coordinator.hass.async_add_executor_job(
            coordinator.database.end_experiment, exp_id, result
        )
        _update_progress(coordinator, 1.0)
        logger.info("PRBS experiment completed: %d switches", switches)

    except asyncio.CancelledError:
        await coordinator.hass.async_add_executor_job(
            coordinator.database.end_experiment, exp_id, {"status": "cancelled"}
        )
        raise


# =============================================================================
# Relay Feedback
# =============================================================================


async def _run_relay_feedback(
    coordinator: RadiatorControlCoordinator,
    high_offset: float = -3.0,
    low_offset: float = 3.0,
    hysteresis: float = RELAY_HYSTERESIS,
    max_duration: int = RELAY_MAX_DURATION,
    min_oscillations: int = 3,
) -> None:
    """Run a relay-feedback experiment."""
    # Get current target temp from thermostat
    data = await _read_sensor(coordinator)
    target_temp = data["target_temp"]

    exp_id = await coordinator.hass.async_add_executor_job(
        coordinator.database.start_experiment,
        f"Relay-Feedback @ {target_temp}°C",
        "relay",
        {
            "target_temp": target_temp,
            "high_offset": high_offset,
            "low_offset": low_offset,
            "hysteresis": hysteresis,
            "max_duration": max_duration,
        },
    )

    temps: list[float] = []
    times: list[float] = []
    crossings: list[float] = []

    try:
        logger.info(
            "Relay-feedback experiment started (target=%.1f°C)", target_temp
        )
        start_time = datetime.now()
        current_offset = high_offset
        elapsed_minutes = 0.0

        while elapsed_minutes < max_duration:
            data = await _collect_and_store(coordinator, current_offset)

            if data["window_open"]:
                logger.warning("Window opened during relay - aborting")
                raise asyncio.CancelledError("Window opened")

            elapsed_minutes = (
                datetime.now() - start_time
            ).total_seconds() / 60
            times.append(elapsed_minutes)
            temps.append(data["room_temp"])

            # Relay logic
            error = data["room_temp"] - target_temp
            if current_offset == high_offset and error > hysteresis:
                current_offset = low_offset
                crossings.append(elapsed_minutes)
            elif current_offset == low_offset and error < -hysteresis:
                current_offset = high_offset
                crossings.append(elapsed_minutes)

            # Progress
            oscillations = len(crossings) // 2
            progress = min(
                oscillations / min_oscillations,
                elapsed_minutes / max_duration,
            )
            _update_progress(coordinator, min(progress, 0.99))

            if len(crossings) >= min_oscillations * 2 + 1:
                logger.info("Enough oscillations measured")
                break

            await asyncio.sleep(SAMPLE_INTERVAL)

        # Analysis
        _update_progress(coordinator, 0.95)
        temps_arr = np.array(temps)

        if len(crossings) < 4:
            logger.warning("Not enough oscillations for relay-feedback tuning")
            result = {"status": "insufficient_data"}
        else:
            periods = np.diff(crossings)
            Tu = float(np.mean(periods) * 2)
            amplitude_relay = abs(high_offset - low_offset) / 2
            amplitude_temp = float(
                (np.max(temps_arr[-len(temps_arr) // 2 :])
                 - np.min(temps_arr[-len(temps_arr) // 2 :]))
                / 2
            )
            Ku = (
                4 * amplitude_relay / (np.pi * amplitude_temp)
                if amplitude_temp > 0.01
                else 1.0
            )

            result = {
                "status": "success",
                "Ku": Ku,
                "Tu": Tu,
                "amplitude_temp": amplitude_temp,
                "n_oscillations": len(crossings) // 2,
                "pid_params": {
                    "PID": {
                        "Kp": round(0.6 * Ku, 3),
                        "Ti": round(Tu / 2, 1),
                        "Td": round(Tu / 8, 1),
                    },
                },
            }
            logger.info("Relay-feedback result: Ku=%.2f, Tu=%.1f min", Ku, Tu)

        await coordinator.hass.async_add_executor_job(
            coordinator.database.end_experiment, exp_id, result
        )
        _update_progress(coordinator, 1.0)

    except asyncio.CancelledError:
        await coordinator.hass.async_add_executor_job(
            coordinator.database.end_experiment, exp_id, {"status": "cancelled"}
        )
        raise
