"""Climate platform for Radiator Control."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.climate import (
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, CONF_THERMOSTAT_ENTITY, MANUFACTURER
from .coordinator import RadiatorControlCoordinator, RadiatorControlData

logger = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up climate entity."""
    coordinator: RadiatorControlCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([RadiatorControlClimate(coordinator, entry)])


class RadiatorControlClimate(
    CoordinatorEntity[RadiatorControlCoordinator], ClimateEntity
):
    """Climate entity representing the MPC-controlled radiator."""

    _attr_has_entity_name = True
    _attr_name = None  # Use device name
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_target_temperature_step = 0.5
    _attr_min_temp = 5.0
    _attr_max_temp = 30.0
    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]
    _attr_supported_features = ClimateEntityFeature.TARGET_TEMPERATURE

    def __init__(
        self,
        coordinator: RadiatorControlCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the climate entity."""
        super().__init__(coordinator)
        self._entry = entry
        thermostat_id = entry.data[CONF_THERMOSTAT_ENTITY]
        self._attr_unique_id = f"{entry.entry_id}_climate"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": f"Radiator Control ({thermostat_id.split('.')[-1]})",
            "manufacturer": MANUFACTURER,
            "model": "MPC v2.0",
            "sw_version": "2.0.0",
        }

    @property
    def current_temperature(self) -> float | None:
        """Return the current temperature."""
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return None
        return data.room_temp

    @property
    def target_temperature(self) -> float | None:
        """Return the target temperature."""
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return None
        return data.target_temp

    @property
    def hvac_mode(self) -> HVACMode:
        """Return current HVAC mode."""
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return HVACMode.OFF
        if data.hvac_mode == "off":
            return HVACMode.OFF
        return HVACMode.HEAT

    @property
    def hvac_action(self) -> HVACAction:
        """Return current HVAC action."""
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return HVACAction.IDLE
        if data.hvac_mode == "off":
            return HVACAction.OFF
        if data.heating_active:
            return HVACAction.HEATING
        return HVACAction.IDLE

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra state attributes."""
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return {}
        return {
            "control_offset": round(data.offset, 2),
            "control_mode": data.mode,
            "control_enabled": data.control_enabled,
            "model_rmse": round(data.model_rmse, 4) if data.model_rmse != float("inf") else None,
            "model_tau": round(data.model_tau, 1),
            "window_open": data.window_open,
            "outside_temperature": data.outside_temp,
        }

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature — forwarded to the real thermostat."""
        temp = kwargs.get(ATTR_TEMPERATURE)
        if temp is None:
            return
        thermostat = self._entry.data[CONF_THERMOSTAT_ENTITY]
        await self.hass.services.async_call(
            "climate",
            "set_temperature",
            {"entity_id": thermostat, ATTR_TEMPERATURE: temp},
            blocking=True,
        )
        await self.coordinator.async_request_refresh()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set new HVAC mode — forwarded to the real thermostat."""
        thermostat = self._entry.data[CONF_THERMOSTAT_ENTITY]
        await self.hass.services.async_call(
            "climate",
            "set_hvac_mode",
            {"entity_id": thermostat, "hvac_mode": hvac_mode.value},
            blocking=True,
        )
        await self.coordinator.async_request_refresh()
