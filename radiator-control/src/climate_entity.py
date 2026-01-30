"""
Climate Entity für Home Assistant
=================================
Erstellt eine virtuelle Climate-Entity, die mit better-thermostat-ui-card kompatibel ist.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .ha_client import HomeAssistantClient

logger = logging.getLogger(__name__)


class ClimateEntity:
    """
    Virtuelle Climate-Entity für Home Assistant.
    
    Ermöglicht:
    - Anzeige in Lovelace mit better-thermostat-ui-card
    - Temperatur setzen via HA Frontend
    - HVAC Mode ändern
    """
    
    def __init__(
        self,
        ha_client: 'HomeAssistantClient',
        entity_id: str = "climate.radiator_mpc_control",
        friendly_name: str = "Radiator MPC Control",
    ):
        self.ha_client = ha_client
        self.entity_id = entity_id
        self.friendly_name = friendly_name
        
        # State
        self._current_temp: Optional[float] = None
        self._target_temp: Optional[float] = None
        self._hvac_mode: str = "heat"  # heat, off
        self._hvac_action: str = "idle"  # heating, idle, off
        self._preset_mode: str = "none"  # none, eco, comfort
        
        # Attributes
        self._attributes = {
            "friendly_name": friendly_name,
            "supported_features": 1,  # SUPPORT_TARGET_TEMPERATURE
            "hvac_modes": ["heat", "off"],
            "min_temp": 5,
            "max_temp": 30,
            "target_temp_step": 0.5,
            "preset_modes": ["none", "eco", "comfort"],
            "icon": "mdi:radiator",
        }
        
        # Extra Attributes für das Add-on
        self._extra_attrs = {
            "control_offset": 0,
            "model_rmse": 0,
            "window_open": False,
        }
    
    async def update(
        self,
        current_temp: Optional[float] = None,
        target_temp: Optional[float] = None,
        hvac_mode: Optional[str] = None,
        hvac_action: Optional[str] = None,
        control_offset: Optional[float] = None,
        model_rmse: Optional[float] = None,
        window_open: Optional[bool] = None,
    ):
        """
        Aktualisiert die Climate-Entity in Home Assistant.
        """
        if current_temp is not None:
            self._current_temp = current_temp
        if target_temp is not None:
            self._target_temp = target_temp
        if hvac_mode is not None:
            self._hvac_mode = hvac_mode
        if hvac_action is not None:
            self._hvac_action = hvac_action
        if control_offset is not None:
            self._extra_attrs["control_offset"] = round(control_offset, 2)
        if model_rmse is not None:
            self._extra_attrs["model_rmse"] = round(model_rmse, 4)
        if window_open is not None:
            self._extra_attrs["window_open"] = window_open
        
        # Build attributes
        attrs = {
            **self._attributes,
            **self._extra_attrs,
            "current_temperature": self._current_temp,
            "temperature": self._target_temp,
            "hvac_action": self._hvac_action,
            "hvac_mode": self._hvac_mode,
            "preset_mode": self._preset_mode,
        }
        
        # Publish to HA
        await self.ha_client.set_state(
            entity_id=self.entity_id,
            state=self._hvac_mode,
            attributes=attrs,
        )
        
        logger.debug(f"Updated climate entity {self.entity_id}")
    
    async def watch_for_changes(self, callback):
        """
        Überwacht Änderungen an der Climate-Entity via WebSocket.
        
        Wenn der Benutzer im HA Frontend die Temperatur ändert,
        wird der Callback aufgerufen.
        
        Args:
            callback: Async Funktion mit (attribute, value) als Argumente
        """
        # Das erfordert WebSocket-Verbindung zu HA
        # Für jetzt nutzen wir Polling als Fallback
        
        last_target = self._target_temp
        last_mode = self._hvac_mode
        
        while True:
            try:
                state = await self.ha_client.get_state(self.entity_id)
                if state:
                    new_target = state.get('attributes', {}).get('temperature')
                    new_mode = state.get('state')
                    
                    if new_target and new_target != last_target:
                        await callback('target_temp', new_target)
                        last_target = new_target
                    
                    if new_mode and new_mode != last_mode:
                        await callback('hvac_mode', new_mode)
                        last_mode = new_mode
                
            except Exception as e:
                logger.debug(f"Error watching climate entity: {e}")
            
            await asyncio.sleep(5)  # Poll alle 5 Sekunden


# =============================================================================
# Helper Functions
# =============================================================================

async def create_climate_entity(
    ha_client: 'HomeAssistantClient',
    entity_id: str = "climate.radiator_mpc_control",
    friendly_name: str = "Radiator MPC Control",
) -> ClimateEntity:
    """
    Erstellt eine neue Climate-Entity in Home Assistant.
    
    Args:
        ha_client: Home Assistant Client
        entity_id: Entity ID für die Climate-Entity
        friendly_name: Anzeigename
    
    Returns:
        ClimateEntity Instanz
    """
    entity = ClimateEntity(
        ha_client=ha_client,
        entity_id=entity_id,
        friendly_name=friendly_name,
    )
    
    # Initial state
    await entity.update(
        current_temp=20.0,
        target_temp=21.0,
        hvac_mode="off",
        hvac_action="idle",
    )
    
    logger.info(f"Created climate entity: {entity_id}")
    return entity
