"""Switch platform for Radiator Control."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchDeviceClass, SwitchEntity
from homeassistant.config_entries import ConfigEntry
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
    """Set up switch entities."""
    coordinator: RadiatorControlCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([RadiatorControlSwitch(coordinator, entry)])


class RadiatorControlSwitch(
    CoordinatorEntity[RadiatorControlCoordinator], SwitchEntity
):
    """Switch to enable/disable MPC control."""

    _attr_has_entity_name = True
    _attr_name = "Control Active"
    _attr_icon = "mdi:radiator"
    _attr_device_class = SwitchDeviceClass.SWITCH

    def __init__(
        self,
        coordinator: RadiatorControlCoordinator,
        entry: ConfigEntry,
    ) -> None:
        super().__init__(coordinator)
        thermostat_id = entry.data[CONF_THERMOSTAT_ENTITY]
        self._attr_unique_id = f"{entry.entry_id}_control_active"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": f"Radiator Control ({thermostat_id.split('.')[-1]})",
        }

    @property
    def is_on(self) -> bool:
        """Return true if control is active."""
        data: RadiatorControlData = self.coordinator.data
        if data is None:
            return False
        return data.control_enabled

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on MPC control."""
        self.coordinator.set_control_enabled(True)
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off MPC control."""
        self.coordinator.set_control_enabled(False)
        self.async_write_ha_state()
