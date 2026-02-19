"""Config flow for Radiator Control MPC integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, OptionsFlow
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_THERMOSTAT_ENTITY,
    CONF_TEMP_SENSOR_ENTITY,
    CONF_WINDOW_SENSOR_ENTITY,
    CONF_OUTSIDE_TEMP_ENTITY,
    CONF_TEMP_CALIBRATION_ENTITY,
    CONF_MPC_HORIZON,
    CONF_MPC_WEIGHT_COMFORT,
    CONF_MPC_WEIGHT_ENERGY,
    CONF_MPC_WEIGHT_SMOOTHNESS,
    CONF_OFFSET_MIN,
    CONF_OFFSET_MAX,
    CONF_SAMPLE_TIME,
    CONF_WINDOW_ACTION,
    CONF_WINDOW_OFF_DELAY,
    CONF_MODEL_FORGETTING_FACTOR,
    DEFAULT_MPC_HORIZON,
    DEFAULT_MPC_WEIGHT_COMFORT,
    DEFAULT_MPC_WEIGHT_ENERGY,
    DEFAULT_MPC_WEIGHT_SMOOTHNESS,
    DEFAULT_OFFSET_MIN,
    DEFAULT_OFFSET_MAX,
    DEFAULT_SAMPLE_TIME,
    DEFAULT_WINDOW_ACTION,
    DEFAULT_WINDOW_OFF_DELAY,
    DEFAULT_MODEL_FORGETTING_FACTOR,
    WINDOW_ACTION_TURN_OFF,
    WINDOW_ACTION_OFFSET,
)


class RadiatorControlConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Radiator Control."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate thermostat entity exists
            thermostat = user_input[CONF_THERMOSTAT_ENTITY]
            state = self.hass.states.get(thermostat)
            if state is None:
                errors[CONF_THERMOSTAT_ENTITY] = "entity_not_found"
            else:
                # Use thermostat name as title
                name = state.attributes.get("friendly_name", thermostat)
                title = f"Radiator Control - {name}"

                # Check if already configured for this thermostat
                await self.async_set_unique_id(thermostat)
                self._abort_if_unique_id_configured()

                return self.async_create_entry(title=title, data=user_input)

        schema = vol.Schema(
            {
                vol.Required(CONF_THERMOSTAT_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="climate")
                ),
                vol.Optional(CONF_TEMP_SENSOR_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor", device_class="temperature")
                ),
                vol.Optional(CONF_WINDOW_SENSOR_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="binary_sensor")
                ),
                vol.Optional(CONF_OUTSIDE_TEMP_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="sensor", device_class="temperature")
                ),
                vol.Optional(CONF_TEMP_CALIBRATION_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain="number")
                ),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return RadiatorControlOptionsFlow(config_entry)


class RadiatorControlOptionsFlow(OptionsFlow):
    """Handle options for Radiator Control."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = self.config_entry.options

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_MPC_HORIZON,
                    default=options.get(CONF_MPC_HORIZON, DEFAULT_MPC_HORIZON),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=60, max=480, step=30, mode="slider",
                        unit_of_measurement="min",
                    )
                ),
                vol.Optional(
                    CONF_MPC_WEIGHT_COMFORT,
                    default=options.get(CONF_MPC_WEIGHT_COMFORT, DEFAULT_MPC_WEIGHT_COMFORT),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.1, max=10.0, step=0.1, mode="slider",
                    )
                ),
                vol.Optional(
                    CONF_MPC_WEIGHT_ENERGY,
                    default=options.get(CONF_MPC_WEIGHT_ENERGY, DEFAULT_MPC_WEIGHT_ENERGY),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0, max=5.0, step=0.05, mode="slider",
                    )
                ),
                vol.Optional(
                    CONF_MPC_WEIGHT_SMOOTHNESS,
                    default=options.get(CONF_MPC_WEIGHT_SMOOTHNESS, DEFAULT_MPC_WEIGHT_SMOOTHNESS),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0, max=1.0, step=0.01, mode="slider",
                    )
                ),
                vol.Optional(
                    CONF_OFFSET_MIN,
                    default=options.get(CONF_OFFSET_MIN, DEFAULT_OFFSET_MIN),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=-10.0, max=0.0, step=0.5, mode="slider",
                        unit_of_measurement="°C",
                    )
                ),
                vol.Optional(
                    CONF_OFFSET_MAX,
                    default=options.get(CONF_OFFSET_MAX, DEFAULT_OFFSET_MAX),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0, max=10.0, step=0.5, mode="slider",
                        unit_of_measurement="°C",
                    )
                ),
                vol.Optional(
                    CONF_SAMPLE_TIME,
                    default=options.get(CONF_SAMPLE_TIME, DEFAULT_SAMPLE_TIME),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=30, max=300, step=10, mode="slider",
                        unit_of_measurement="s",
                    )
                ),
                vol.Optional(
                    CONF_WINDOW_ACTION,
                    default=options.get(CONF_WINDOW_ACTION, DEFAULT_WINDOW_ACTION),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(value=WINDOW_ACTION_TURN_OFF, label="Thermostat ausschalten"),
                            selector.SelectOptionDict(value=WINDOW_ACTION_OFFSET, label="Nur Offset maximieren"),
                        ],
                        mode="dropdown",
                    )
                ),
                vol.Optional(
                    CONF_WINDOW_OFF_DELAY,
                    default=options.get(CONF_WINDOW_OFF_DELAY, DEFAULT_WINDOW_OFF_DELAY),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=300, step=10, mode="slider",
                        unit_of_measurement="s",
                    )
                ),
                vol.Optional(
                    CONF_MODEL_FORGETTING_FACTOR,
                    default=options.get(CONF_MODEL_FORGETTING_FACTOR, DEFAULT_MODEL_FORGETTING_FACTOR),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.9, max=0.999, step=0.001, mode="box",
                    )
                ),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
