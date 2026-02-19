"""Number platform for Radiator Control."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.number import NumberDeviceClass, NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    CONF_THERMOSTAT_ENTITY,
    CONF_MPC_HORIZON,
    CONF_MPC_WEIGHT_COMFORT,
    CONF_MPC_WEIGHT_ENERGY,
    CONF_MPC_WEIGHT_SMOOTHNESS,
    CONF_OFFSET_MIN,
    CONF_OFFSET_MAX,
    DEFAULT_MPC_HORIZON,
    DEFAULT_MPC_WEIGHT_COMFORT,
    DEFAULT_MPC_WEIGHT_ENERGY,
    DEFAULT_MPC_WEIGHT_SMOOTHNESS,
    DEFAULT_OFFSET_MIN,
    DEFAULT_OFFSET_MAX,
)
from .coordinator import RadiatorControlCoordinator

logger = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up number entities."""
    coordinator: RadiatorControlCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([
        MPCHorizonNumber(coordinator, entry),
        WeightComfortNumber(coordinator, entry),
        WeightEnergyNumber(coordinator, entry),
        WeightSmoothnessNumber(coordinator, entry),
        OffsetMinNumber(coordinator, entry),
        OffsetMaxNumber(coordinator, entry),
    ])


def _device_info(entry: ConfigEntry) -> dict:
    thermostat_id = entry.data[CONF_THERMOSTAT_ENTITY]
    return {
        "identifiers": {(DOMAIN, entry.entry_id)},
        "name": f"Radiator Control ({thermostat_id.split('.')[-1]})",
    }


class _BaseNumber(
    CoordinatorEntity[RadiatorControlCoordinator], NumberEntity
):
    """Base class for Radiator Control number entities."""

    _attr_has_entity_name = True
    _config_key: str = ""
    _default_value: float = 0.0

    def __init__(
        self,
        coordinator: RadiatorControlCoordinator,
        entry: ConfigEntry,
    ) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_device_info = _device_info(entry)

    @property
    def native_value(self) -> float:
        return self.coordinator.entry.options.get(
            self._config_key,
            self.coordinator.entry.data.get(self._config_key, self._default_value),
        )

    async def async_set_native_value(self, value: float) -> None:
        """Update the value and store in options."""
        new_options = dict(self.coordinator.entry.options)
        new_options[self._config_key] = value
        self.hass.config_entries.async_update_entry(
            self.coordinator.entry, options=new_options
        )
        # Rebuild MPC with new parameters
        self.coordinator.rebuild_mpc()
        self.async_write_ha_state()


class MPCHorizonNumber(_BaseNumber):
    """Number entity for MPC prediction horizon."""

    _attr_name = "MPC Horizon"
    _attr_icon = "mdi:timer-outline"
    _attr_native_min_value = 60
    _attr_native_max_value = 480
    _attr_native_step = 30
    _attr_mode = NumberMode.SLIDER
    _attr_native_unit_of_measurement = "min"
    _config_key = CONF_MPC_HORIZON
    _default_value = DEFAULT_MPC_HORIZON

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_mpc_horizon"


class WeightComfortNumber(_BaseNumber):
    """Number entity for comfort weight."""

    _attr_name = "Weight Comfort"
    _attr_icon = "mdi:sofa"
    _attr_native_min_value = 0.1
    _attr_native_max_value = 10.0
    _attr_native_step = 0.1
    _attr_mode = NumberMode.SLIDER
    _config_key = CONF_MPC_WEIGHT_COMFORT
    _default_value = DEFAULT_MPC_WEIGHT_COMFORT

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_weight_comfort"


class WeightEnergyNumber(_BaseNumber):
    """Number entity for energy weight."""

    _attr_name = "Weight Energy"
    _attr_icon = "mdi:flash"
    _attr_native_min_value = 0.0
    _attr_native_max_value = 5.0
    _attr_native_step = 0.05
    _attr_mode = NumberMode.SLIDER
    _config_key = CONF_MPC_WEIGHT_ENERGY
    _default_value = DEFAULT_MPC_WEIGHT_ENERGY

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_weight_energy"


class WeightSmoothnessNumber(_BaseNumber):
    """Number entity for smoothness weight."""

    _attr_name = "Weight Smoothness"
    _attr_icon = "mdi:chart-bell-curve"
    _attr_native_min_value = 0.0
    _attr_native_max_value = 1.0
    _attr_native_step = 0.01
    _attr_mode = NumberMode.SLIDER
    _config_key = CONF_MPC_WEIGHT_SMOOTHNESS
    _default_value = DEFAULT_MPC_WEIGHT_SMOOTHNESS

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_weight_smoothness"


class OffsetMinNumber(_BaseNumber):
    """Number entity for minimum offset."""

    _attr_name = "Offset Min"
    _attr_icon = "mdi:thermometer-minus"
    _attr_native_min_value = -10.0
    _attr_native_max_value = 0.0
    _attr_native_step = 0.5
    _attr_mode = NumberMode.SLIDER
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _config_key = CONF_OFFSET_MIN
    _default_value = DEFAULT_OFFSET_MIN

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_offset_min"


class OffsetMaxNumber(_BaseNumber):
    """Number entity for maximum offset."""

    _attr_name = "Offset Max"
    _attr_icon = "mdi:thermometer-plus"
    _attr_native_min_value = 0.0
    _attr_native_max_value = 10.0
    _attr_native_step = 0.5
    _attr_mode = NumberMode.SLIDER
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _config_key = CONF_OFFSET_MAX
    _default_value = DEFAULT_OFFSET_MAX

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator, entry)
        self._attr_unique_id = f"{entry.entry_id}_offset_max"
