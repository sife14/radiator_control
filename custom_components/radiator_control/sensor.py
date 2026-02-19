"""Sensor platform for Radiator Control."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, CONF_THERMOSTAT_ENTITY
from .coordinator import RadiatorControlCoordinator, RadiatorControlData

logger = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensor entities."""
    coordinator: RadiatorControlCoordinator = hass.data[DOMAIN][entry.entry_id]

    sensors = [
        RadiatorControlOffsetSensor(coordinator, entry),
        RadiatorControlModeSensor(coordinator, entry),
        RadiatorControlRMSESensor(coordinator, entry),
        RadiatorControlModelTauSensor(coordinator, entry),
        RadiatorControlModelKHeaterSensor(coordinator, entry),
        RadiatorControlModelUpdatesSensor(coordinator, entry),
        RadiatorControlMPCCostSensor(coordinator, entry),
    ]
    async_add_entities(sensors)


def _device_info(entry: ConfigEntry) -> dict[str, Any]:
    """Return shared device info."""
    thermostat_id = entry.data[CONF_THERMOSTAT_ENTITY]
    return {
        "identifiers": {(DOMAIN, entry.entry_id)},
        "name": f"Radiator Control ({thermostat_id.split('.')[-1]})",
    }


class RadiatorControlOffsetSensor(
    CoordinatorEntity[RadiatorControlCoordinator], SensorEntity
):
    """Sensor showing the current temperature offset."""

    _attr_has_entity_name = True
    _attr_name = "Control Offset"
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:thermometer-plus"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_offset"
        self._attr_device_info = _device_info(entry)

    @property
    def native_value(self) -> float | None:
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return None
        return round(data.offset, 2)


class RadiatorControlModeSensor(
    CoordinatorEntity[RadiatorControlCoordinator], SensorEntity
):
    """Sensor showing the current control mode."""

    _attr_has_entity_name = True
    _attr_name = "Control Mode"
    _attr_icon = "mdi:radiator"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_mode"
        self._attr_device_info = _device_info(entry)

    @property
    def native_value(self) -> str | None:
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return None
        return data.mode

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return {}
        attrs: dict[str, Any] = {"control_enabled": data.control_enabled}
        if data.experiment_type:
            attrs["experiment_type"] = data.experiment_type
            attrs["experiment_progress"] = round(data.experiment_progress * 100, 1)
        return attrs


class RadiatorControlRMSESensor(
    CoordinatorEntity[RadiatorControlCoordinator], SensorEntity
):
    """Sensor showing the model RMSE."""

    _attr_has_entity_name = True
    _attr_name = "Model RMSE"
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-line"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_rmse"
        self._attr_device_info = _device_info(entry)

    @property
    def native_value(self) -> float | None:
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return None
        rmse = data.model_rmse
        if rmse == float("inf"):
            return None
        return round(rmse, 4)


class RadiatorControlModelTauSensor(
    CoordinatorEntity[RadiatorControlCoordinator], SensorEntity
):
    """Sensor showing the model time constant tau."""

    _attr_has_entity_name = True
    _attr_name = "Model τ"
    _attr_native_unit_of_measurement = "min"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:timer-sand"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_model_tau"
        self._attr_device_info = _device_info(entry)

    @property
    def native_value(self) -> float | None:
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return None
        return round(data.model_tau, 1)


class RadiatorControlModelKHeaterSensor(
    CoordinatorEntity[RadiatorControlCoordinator], SensorEntity
):
    """Sensor showing the model heater gain."""

    _attr_has_entity_name = True
    _attr_name = "Model K_heater"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:fire"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_model_k_heater"
        self._attr_device_info = _device_info(entry)

    @property
    def native_value(self) -> float | None:
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return None
        return round(data.model_k_heater, 3)


class RadiatorControlModelUpdatesSensor(
    CoordinatorEntity[RadiatorControlCoordinator], SensorEntity
):
    """Sensor showing the number of model updates."""

    _attr_has_entity_name = True
    _attr_name = "Model Updates"
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_icon = "mdi:counter"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_model_updates"
        self._attr_device_info = _device_info(entry)

    @property
    def native_value(self) -> int | None:
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return None
        return data.model_updates


class RadiatorControlMPCCostSensor(
    CoordinatorEntity[RadiatorControlCoordinator], SensorEntity
):
    """Sensor showing the MPC cost value."""

    _attr_has_entity_name = True
    _attr_name = "MPC Cost"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-bar"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_mpc_cost"
        self._attr_device_info = _device_info(entry)

    @property
    def native_value(self) -> float | None:
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return None
        cost = data.mpc_cost
        if cost == float("inf"):
            return None
        return round(cost, 2)
