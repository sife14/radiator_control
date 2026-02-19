"""Button platform for Radiator Control."""

from __future__ import annotations

import logging

from homeassistant.components.button import ButtonDeviceClass, ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, CONF_THERMOSTAT_ENTITY
from .coordinator import RadiatorControlCoordinator

logger = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up button entities."""
    coordinator: RadiatorControlCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([
        ExperimentStepButton(coordinator, entry),
        ExperimentPRBSButton(coordinator, entry),
        ExperimentRelayButton(coordinator, entry),
        StopExperimentButton(coordinator, entry),
        ResetModelButton(coordinator, entry),
    ])


def _device_info(entry: ConfigEntry) -> dict:
    thermostat_id = entry.data[CONF_THERMOSTAT_ENTITY]
    return {
        "identifiers": {(DOMAIN, entry.entry_id)},
        "name": f"Radiator Control ({thermostat_id.split('.')[-1]})",
    }


class ExperimentStepButton(
    CoordinatorEntity[RadiatorControlCoordinator], ButtonEntity
):
    """Button to start a step response experiment."""

    _attr_has_entity_name = True
    _attr_name = "Experiment: Step Response"
    _attr_icon = "mdi:chart-bell-curve-cumulative"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_exp_step"
        self._attr_device_info = _device_info(entry)

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.start_experiment("step")


class ExperimentPRBSButton(
    CoordinatorEntity[RadiatorControlCoordinator], ButtonEntity
):
    """Button to start a PRBS experiment."""

    _attr_has_entity_name = True
    _attr_name = "Experiment: PRBS"
    _attr_icon = "mdi:chart-scatter-plot"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_exp_prbs"
        self._attr_device_info = _device_info(entry)

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.start_experiment("prbs")


class ExperimentRelayButton(
    CoordinatorEntity[RadiatorControlCoordinator], ButtonEntity
):
    """Button to start a relay feedback experiment."""

    _attr_has_entity_name = True
    _attr_name = "Experiment: Relay Feedback"
    _attr_icon = "mdi:sine-wave"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_exp_relay"
        self._attr_device_info = _device_info(entry)

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.start_experiment("relay")


class StopExperimentButton(
    CoordinatorEntity[RadiatorControlCoordinator], ButtonEntity
):
    """Button to stop a running experiment."""

    _attr_has_entity_name = True
    _attr_name = "Stop Experiment"
    _attr_icon = "mdi:stop-circle"

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_exp_stop"
        self._attr_device_info = _device_info(entry)

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.stop_experiment()


class ResetModelButton(
    CoordinatorEntity[RadiatorControlCoordinator], ButtonEntity
):
    """Button to reset the thermal model."""

    _attr_has_entity_name = True
    _attr_name = "Reset Model"
    _attr_icon = "mdi:refresh"
    _attr_device_class = ButtonDeviceClass.RESTART

    def __init__(self, coordinator: RadiatorControlCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_reset_model"
        self._attr_device_info = _device_info(entry)

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.hass.async_add_executor_job(self.coordinator.reset_model)
