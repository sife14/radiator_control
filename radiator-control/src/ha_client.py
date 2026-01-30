"""
Home Assistant Client
=====================
Kommunikation mit Home Assistant via REST API und WebSocket.
"""

import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Container für alle Sensordaten."""
    timestamp: datetime
    room_temp: float
    outside_temp: Optional[float]
    window_open: bool
    thermostat_setpoint: float
    heating_active: Optional[bool]
    hvac_mode: Optional[str] = None  # 'heat', 'off', 'auto'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'room_temp': self.room_temp,
            'outside_temp': self.outside_temp,
            'window_open': self.window_open,
            'thermostat_setpoint': self.thermostat_setpoint,
            'heating_active': self.heating_active,
            'hvac_mode': self.hvac_mode,
        }


class HomeAssistantClient:
    """
    Client für Home Assistant Kommunikation.
    
    Unterstützt:
    - Lesen von Sensorwerten
    - Schreiben von Temperaturen an Thermostat
    - WebSocket für Echtzeit-Updates
    """
    
    def __init__(self, url: str, token: str, entities: Dict[str, str]):
        """
        Args:
            url: Home Assistant URL (z.B. "http://homeassistant.local:8123")
            token: Long-Lived Access Token
            entities: Dict mit Entity-IDs für thermostat, temp_sensor, etc.
        """
        self.url = url.rstrip('/')
        self.token = token
        self.entities = entities
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_id = 0
        
    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
    
    async def connect(self):
        """Verbindung herstellen."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        
        # Test connection
        try:
            async with self._session.get(
                f"{self.url}/api/",
                headers=self.headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Connected to Home Assistant: {data.get('message', 'OK')}")
                else:
                    raise ConnectionError(f"HA returned status {resp.status}")
        except Exception as e:
            logger.error(f"Failed to connect to Home Assistant: {e}")
            raise
    
    async def disconnect(self):
        """Verbindung trennen."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._session:
            await self._session.close()
            self._session = None
    
    async def get_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Zustand einer Entity abfragen.
        
        Returns:
            Dict mit 'state' und 'attributes' oder None bei Fehler.
        """
        if not self._session:
            await self.connect()
        
        try:
            async with self._session.get(
                f"{self.url}/api/states/{entity_id}",
                headers=self.headers
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 404:
                    logger.warning(f"Entity not found: {entity_id}")
                    return None
                else:
                    logger.error(f"Error getting state for {entity_id}: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Exception getting state for {entity_id}: {e}")
            return None
    
    async def get_sensor_value(self, entity_id: str) -> Optional[float]:
        """Numerischen Sensorwert abrufen."""
        state = await self.get_state(entity_id)
        if state and state.get('state') not in ('unknown', 'unavailable', None):
            try:
                return float(state['state'])
            except (ValueError, TypeError):
                logger.warning(f"Cannot convert state to float: {state['state']}")
        return None
    
    async def get_binary_sensor(self, entity_id: str) -> Optional[bool]:
        """Binären Sensorwert abrufen (on/off)."""
        state = await self.get_state(entity_id)
        if state:
            return state.get('state') == 'on'
        return None
    
    async def read_all_sensors(self) -> SensorData:
        """
        Alle relevanten Sensoren auslesen.
        
        Returns:
            SensorData mit allen aktuellen Werten.
        """
        # Parallel alle Sensoren abfragen
        tasks = {}
        
        # Temperatur: Entweder separater Sensor oder vom Thermostat
        if self.entities.get('temp_sensor'):
            tasks['room_temp'] = self.get_sensor_value(self.entities['temp_sensor'])
        else:
            tasks['room_temp'] = self._get_thermostat_current_temp()
        
        if self.entities.get('outside_temp'):
            tasks['outside_temp'] = self.get_sensor_value(self.entities['outside_temp'])
        
        if self.entities.get('window_sensor'):
            tasks['window_open'] = self.get_binary_sensor(self.entities['window_sensor'])
        
        tasks['thermostat_state'] = self.get_state(self.entities['thermostat'])
        
        # Alle parallel ausführen
        results = {}
        for key, coro in tasks.items():
            results[key] = await coro
        
        # Thermostat-Daten extrahieren
        thermo_state = results.get('thermostat_state', {})
        thermo_attrs = thermo_state.get('attributes', {}) if thermo_state else {}
        
        # Heizstatus aus Thermostat-Attributen oder separatem Sensor
        heating_active = None
        if self.entities.get('heating_state'):
            heating_active = await self.get_binary_sensor(self.entities['heating_state'])
        elif thermo_attrs.get('hvac_action'):
            heating_active = thermo_attrs['hvac_action'] == 'heating'
        
        # HVAC Mode (off, heat, auto, etc.)
        hvac_mode = thermo_state.get('state') if thermo_state else None
        
        return SensorData(
            timestamp=datetime.now(),
            room_temp=results.get('room_temp', 20.0),
            outside_temp=results.get('outside_temp'),
            window_open=results.get('window_open', False),
            thermostat_setpoint=thermo_attrs.get('temperature', 21.0),
            heating_active=heating_active,
            hvac_mode=hvac_mode,
        )
    
    async def _get_thermostat_current_temp(self) -> Optional[float]:
        """Aktuelle Temperatur vom Thermostat lesen."""
        state = await self.get_state(self.entities['thermostat'])
        if state:
            attrs = state.get('attributes', {})
            return attrs.get('current_temperature')
        return None
    
    async def set_thermostat_temperature(self, temperature: float):
        """
        Solltemperatur am Thermostat setzen.
        
        Args:
            temperature: Neue Solltemperatur
        """
        if not self._session:
            await self.connect()
        
        service_data = {
            "entity_id": self.entities['thermostat'],
            "temperature": round(temperature, 1),
        }
        
        try:
            async with self._session.post(
                f"{self.url}/api/services/climate/set_temperature",
                headers=self.headers,
                json=service_data
            ) as resp:
                if resp.status == 200:
                    logger.debug(f"Set thermostat to {temperature}°C")
                else:
                    logger.error(f"Failed to set temperature: {resp.status}")
        except Exception as e:
            logger.error(f"Exception setting temperature: {e}")
    
    async def set_hvac_mode(self, mode: str):
        """
        HVAC-Modus setzen (heat, off, auto).
        
        Args:
            mode: 'heat', 'off', 'auto'
        """
        if not self._session:
            await self.connect()
        
        service_data = {
            "entity_id": self.entities['thermostat'],
            "hvac_mode": mode,
        }
        
        try:
            async with self._session.post(
                f"{self.url}/api/services/climate/set_hvac_mode",
                headers=self.headers,
                json=service_data
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Set thermostat HVAC mode to {mode}")
                else:
                    logger.error(f"Failed to set HVAC mode: {resp.status}")
        except Exception as e:
            logger.error(f"Exception setting HVAC mode: {e}")
    
    async def turn_off(self):
        """Thermostat ausschalten."""
        await self.set_hvac_mode('off')
    
    async def turn_on(self):
        """Thermostat einschalten (heat mode)."""
        await self.set_hvac_mode('heat')
    
    async def set_fake_temperature(self, fake_temp: float):
        """
        "Fake" Temperatur an Thermostat senden.
        
        Bei Zigbee2MQTT: Setzt local_temperature_calibration so,
        dass das Thermostat die fake_temp "sieht".
        
        Alternative: Direkt via MQTT.
        
        Args:
            fake_temp: Die Temperatur die das Thermostat "sehen" soll.
        """
        # Methode 1: Über Zigbee2MQTT MQTT Topic
        # Dafür brauchen wir die MQTT Integration
        
        # Methode 2: Über HA Service (wenn verfügbar)
        # Silvercrest Thermostate haben oft ein "local_temperature_calibration" Attribut
        
        # Hier: Wir berechnen das nötige Offset
        current_temp = await self._get_thermostat_current_temp()
        if current_temp is None:
            logger.error("Cannot read current temperature from thermostat")
            return
        
        # Offset = was wir wollen - was es misst
        # Wenn wir mehr heizen wollen, sagen wir es ist kälter → negatives Offset
        calibration_offset = fake_temp - current_temp
        
        # Auf ganze oder halbe Grade runden (je nach Thermostat)
        calibration_offset = round(calibration_offset * 2) / 2
        
        # Begrenzen auf sinnvollen Bereich
        calibration_offset = max(-5.0, min(5.0, calibration_offset))
        
        await self._set_temperature_calibration(calibration_offset)
    
    async def set_temperature_offset(self, offset: float):
        """
        Temperatur-Offset direkt setzen.
        
        offset > 0: Thermostat denkt es ist wärmer → weniger heizen
        offset < 0: Thermostat denkt es ist kälter → mehr heizen
        
        Args:
            offset: Offset in °C
        """
        await self._set_temperature_calibration(offset)
    
    async def _set_temperature_calibration(self, offset: float):
        """
        Setzt das local_temperature_calibration Attribut.
        
        Bei Zigbee2MQTT Geräten wird dies typischerweise über
        einen Number-Entity oder MQTT gemacht.
        """
        if not self._session:
            await self.connect()
        
        # Versuch 1: Über number Entity (Zigbee2MQTT erstellt diese oft)
        calibration_entity = self.entities.get('temp_calibration')
        if not calibration_entity:
            # Versuche Standard-Namen
            thermostat_name = self.entities['thermostat'].split('.')[-1]
            calibration_entity = f"number.{thermostat_name}_local_temperature_calibration"
        
        # Prüfen ob Entity existiert
        state = await self.get_state(calibration_entity)
        if state:
            service_data = {
                "entity_id": calibration_entity,
                "value": offset,
            }
            try:
                async with self._session.post(
                    f"{self.url}/api/services/number/set_value",
                    headers=self.headers,
                    json=service_data
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Set temperature calibration to {offset}°C")
                        return
            except Exception as e:
                logger.warning(f"Failed via number entity: {e}")
        
        # Versuch 2: Über Zigbee2MQTT MQTT (falls MQTT Integration aktiv)
        await self._set_calibration_via_mqtt(offset)
    
    async def set_state(
        self, 
        entity_id: str, 
        state: Any, 
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Setzt den State einer Entity.
        
        Nützlich um eigene Sensoren in HA zu erstellen/aktualisieren.
        
        Args:
            entity_id: Entity ID (z.B. sensor.radiator_control_rmse)
            state: Der State-Wert
            attributes: Optionale Attribute (unit_of_measurement, friendly_name, etc.)
        """
        if not self._session:
            await self.connect()
        
        data = {
            "state": state,
            "attributes": attributes or {},
        }
        
        try:
            async with self._session.post(
                f"{self.url}/api/states/{entity_id}",
                headers=self.headers,
                json=data
            ) as resp:
                if resp.status in (200, 201):
                    logger.debug(f"Set state {entity_id} = {state}")
                else:
                    logger.error(f"Failed to set state for {entity_id}: {resp.status}")
        except Exception as e:
            logger.error(f"Exception setting state for {entity_id}: {e}")
    
    async def _set_calibration_via_mqtt(self, offset: float):
        """
        Setzt Kalibrierung direkt via MQTT.
        
        Setzt voraus, dass die MQTT-Integration in HA aktiv ist.
        """
        if not self._session:
            await self.connect()
        
        # Device Name aus Entity-ID extrahieren
        thermostat_name = self.entities['thermostat'].split('.')[-1]
        
        # MQTT Topic für Zigbee2MQTT
        # Typisch: zigbee2mqtt/<device_name>/set
        mqtt_topic = f"zigbee2mqtt/{thermostat_name}/set"
        
        payload = {
            "local_temperature_calibration": offset
        }
        
        service_data = {
            "topic": mqtt_topic,
            "payload": str(payload).replace("'", '"'),
        }
        
        try:
            async with self._session.post(
                f"{self.url}/api/services/mqtt/publish",
                headers=self.headers,
                json=service_data
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Set calibration via MQTT to {offset}°C")
                else:
                    logger.error(f"MQTT publish failed: {resp.status}")
        except Exception as e:
            logger.error(f"Exception in MQTT publish: {e}")
    
    async def turn_off_thermostat(self):
        """Thermostat ausschalten (HVAC mode = off)."""
        service_data = {
            "entity_id": self.entities['thermostat'],
            "hvac_mode": "off",
        }
        await self._call_service("climate", "set_hvac_mode", service_data)
    
    async def turn_on_thermostat(self):
        """Thermostat einschalten (HVAC mode = heat)."""
        service_data = {
            "entity_id": self.entities['thermostat'],
            "hvac_mode": "heat",
        }
        await self._call_service("climate", "set_hvac_mode", service_data)
    
    async def _call_service(self, domain: str, service: str, data: Dict):
        """Generischer Service-Aufruf."""
        if not self._session:
            await self.connect()
        
        try:
            async with self._session.post(
                f"{self.url}/api/services/{domain}/{service}",
                headers=self.headers,
                json=data
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Service call failed: {domain}/{service} = {resp.status}")
        except Exception as e:
            logger.error(f"Exception in service call: {e}")
    
    # -------------------------------------------------------------------------
    # WebSocket für Echtzeit-Updates
    # -------------------------------------------------------------------------
    
    async def subscribe_state_changes(
        self, 
        callback: Callable[[str, Dict], None],
        entity_ids: Optional[list] = None
    ):
        """
        WebSocket-Subscription für State Changes.
        
        Args:
            callback: Funktion die bei Änderungen aufgerufen wird
            entity_ids: Nur diese Entities überwachen (None = alle)
        """
        ws_url = self.url.replace('http', 'ws') + '/api/websocket'
        
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        self._ws = await self._session.ws_connect(ws_url)
        
        # Auth
        msg = await self._ws.receive_json()
        if msg.get('type') == 'auth_required':
            await self._ws.send_json({
                "type": "auth",
                "access_token": self.token
            })
            msg = await self._ws.receive_json()
            if msg.get('type') != 'auth_ok':
                raise ConnectionError("WebSocket authentication failed")
        
        # Subscribe to state_changed events
        self._ws_id += 1
        await self._ws.send_json({
            "id": self._ws_id,
            "type": "subscribe_events",
            "event_type": "state_changed"
        })
        
        logger.info("WebSocket subscription active")
        
        # Message loop
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.json()
                if data.get('type') == 'event':
                    event_data = data.get('event', {}).get('data', {})
                    entity_id = event_data.get('entity_id')
                    
                    if entity_ids is None or entity_id in entity_ids:
                        new_state = event_data.get('new_state', {})
                        callback(entity_id, new_state)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {msg}")
                break


# -----------------------------------------------------------------------------
# Convenience Funktionen
# -----------------------------------------------------------------------------

async def test_connection(config: Dict) -> bool:
    """
    Testet die Verbindung zu Home Assistant.
    
    Args:
        config: Konfiguration mit homeassistant.url und homeassistant.token
        
    Returns:
        True wenn Verbindung erfolgreich
    """
    client = HomeAssistantClient(
        url=config['homeassistant']['url'],
        token=config['homeassistant']['token'],
        entities=config['entities']
    )
    
    try:
        await client.connect()
        sensors = await client.read_all_sensors()
        logger.info(f"Test read successful: {sensors}")
        await client.disconnect()
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    import yaml
    
    # Test
    logging.basicConfig(level=logging.DEBUG)
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    asyncio.run(test_connection(config))
