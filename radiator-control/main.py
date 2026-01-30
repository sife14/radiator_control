#!/usr/bin/env python3
"""
Radiator Control - Hauptprogramm
================================
MPC-basierte Heizungsregelung mit adaptivem Modell.

Verwendung:
    python main.py                    # Normaler Regelbetrieb
    python main.py --experiment step  # Sprungantwort-Experiment
    python main.py --experiment prbs  # PRBS-Experiment
    python main.py --test             # Verbindungstest
"""

import asyncio
import argparse
import signal
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np

from src.utils import load_config, setup_logging, clamp
from src.ha_client import HomeAssistantClient
from src.database import Database, Measurement
from src.model import ThermalModel, ThermalModelParams
from src.mpc_controller import MPCController, MPCConfig
from src.experiments import ExperimentRunner, ExperimentConfig

logger = logging.getLogger(__name__)


class RadiatorController:
    """
    Hauptklasse für die Heizungsregelung.
    
    Kombiniert:
    - Home Assistant Kommunikation
    - Adaptives thermisches Modell
    - MPC Optimierung
    - Datenlogging
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        # Logging einrichten
        log_cfg = self.config.get('logging', {})
        setup_logging(
            level=log_cfg.get('level', 'INFO'),
            log_file=log_cfg.get('file'),
        )
        
        # Komponenten initialisieren
        self.db = Database(self.config['database']['path'])
        
        self.ha_client = HomeAssistantClient(
            url=self.config['homeassistant']['url'],
            token=self.config['homeassistant']['token'],
            entities=self.config['entities'],
        )
        
        # Modell laden oder neu erstellen
        model_path = Path("data/model.json")
        if model_path.exists():
            self.model = ThermalModel.load(str(model_path))
            logger.info("Loaded existing model")
        else:
            model_cfg = self.config.get('model', {})
            self.model = ThermalModel(
                initial_params=ThermalModelParams(
                    tau=model_cfg.get('initial_tau', 120),
                    k_heater=model_cfg.get('initial_k_heater', 0.5),
                ),
                forgetting_factor=model_cfg.get('rls_forgetting_factor', 0.98),
            )
            logger.info("Created new model with default parameters")
        
        # MPC Controller
        mpc_cfg = self.config.get('mpc', {})
        ctrl_cfg = self.config.get('control', {})
        
        self.mpc = MPCController(
            model=self.model,
            config=MPCConfig(
                horizon_steps=mpc_cfg.get('horizon_minutes', 120) // 5,
                control_horizon=mpc_cfg.get('control_horizon', 10),
                dt_minutes=5.0,
                weight_comfort=mpc_cfg.get('weight_comfort', 1.0),
                weight_energy=mpc_cfg.get('weight_energy', 0.1),
                weight_smoothness=mpc_cfg.get('weight_smoothness', 0.05),
                offset_min=ctrl_cfg.get('offset_min', -5.0),
                offset_max=ctrl_cfg.get('offset_max', 5.0),
            ),
        )
        
        # Experiment Runner
        self.experiment_runner = ExperimentRunner(
            ha_client=self.ha_client,
            database=self.db,
        )
        
        # State
        self._running = False
        self._mode = self.config.get('modes', {}).get('default', 'control')
        self._previous_offset = 0.0
        self._previous_temp: Optional[float] = None
        self._previous_time: Optional[datetime] = None
        self._previous_window_open: bool = False
    
    def _calculate_reward(self, current_temp: float, target_temp: float, 
                          offset: float, window_open: bool) -> float:
        """
        Berechnet Reward für RL-Training.
        
        Reward-Struktur:
        - Hoher Reward wenn Temperatur nahe Sollwert
        - Penalty für Energieverbrauch (negativer Offset = heizen)
        - Keine Penalty bei offenem Fenster (nicht beeinflussbar)
        """
        if window_open:
            return 0.0  # Neutral bei offenem Fenster
        
        # Temperatur-Komfort: Gauß-Kurve um Sollwert
        temp_error = abs(current_temp - target_temp)
        comfort_reward = np.exp(-0.5 * (temp_error / 0.5) ** 2)  # σ=0.5°C
        
        # Energie-Penalty: Je mehr geheizt wird, desto mehr Penalty
        # offset < 0 = heizen, offset > 0 = nicht heizen
        energy_penalty = 0.1 * max(0, -offset)  # Nur Penalty fürs Heizen
        
        return comfort_reward - energy_penalty
    
    async def start(self):
        """Startet die Regelung."""
        logger.info("Starting Radiator Controller")
        
        # Verbindung herstellen
        await self.ha_client.connect()
        
        # Solltemperatur kommt aus Home Assistant - nicht mehr hier setzen
        # Der Nutzer stellt die gewünschte Temperatur direkt am Thermostat oder in HA ein
        logger.info("Target temperature will be read from Home Assistant thermostat setpoint")
        
        self._running = True
        
        # Hauptschleife
        await self._control_loop()
    
    async def stop(self):
        """Stoppt die Regelung sauber."""
        logger.info("Stopping Radiator Controller")
        self._running = False
        
        # Offset zurücksetzen
        await self.ha_client.set_temperature_offset(0)
        
        # Modell speichern
        self.model.save("data/model.json")
        
        # Verbindung trennen
        await self.ha_client.disconnect()
    
    async def _control_loop(self):
        """Hauptregelschleife."""
        sample_time = self.config['control'].get('sample_time', 60)
        window_action = self.config['control'].get('window_action', 'turn_off')  # 'turn_off' oder 'offset'
        
        # Merken ob Thermostat wegen Fenster ausgeschaltet wurde
        thermostat_was_on = True
        saved_hvac_mode = 'heat'
        
        while self._running:
            try:
                loop_start = datetime.now()
                
                # Sensoren lesen (inkl. Solltemperatur vom Thermostat)
                sensor_data = await self.ha_client.read_all_sensors()
                current_temp = sensor_data.room_temp
                outside_temp = sensor_data.outside_temp or 5.0
                window_open = sensor_data.window_open
                target_temp = sensor_data.thermostat_setpoint  # Aus Home Assistant!
                hvac_mode = sensor_data.hvac_mode
                
                logger.debug(
                    f"Sensors: T_room={current_temp:.1f}°C, T_target={target_temp:.1f}°C, "
                    f"T_out={outside_temp:.1f}°C, window={'open' if window_open else 'closed'}, "
                    f"hvac_mode={hvac_mode}"
                )
                
                # Fenster offen → Heizung komplett ausschalten
                if window_open:
                    self._mode = 'window_open'
                    
                    if window_action == 'turn_off':
                        # Thermostat komplett ausschalten
                        if thermostat_was_on and hvac_mode != 'off':
                            saved_hvac_mode = hvac_mode or 'heat'
                            await self.ha_client.turn_off()
                            logger.info("Window open - thermostat turned OFF")
                            thermostat_was_on = False
                        new_offset = 0  # Offset irrelevant wenn aus
                    else:
                        # Alte Methode: Offset maximieren
                        new_offset = self.config['control'].get('offset_max', 5.0)
                        logger.info(f"Window open - heating OFF (offset={new_offset}°C)")
                    
                    # Bei offenem Fenster: Modell NICHT updaten (verfälscht Daten)
                else:
                    # Fenster zu → ggf. Thermostat wieder einschalten
                    if not thermostat_was_on:
                        await self.ha_client.set_hvac_mode(saved_hvac_mode)
                        logger.info(f"Window closed - thermostat restored to {saved_hvac_mode}")
                        thermostat_was_on = True
                        # Kurz warten bis Thermostat bereit
                        await asyncio.sleep(2)
                    
                    self._mode = 'control'
                    
                    # Modell updaten (nur bei geschlossenem Fenster!)
                    if self._previous_temp is not None and self._previous_time is not None:
                        dt_minutes = (loop_start - self._previous_time).total_seconds() / 60
                        
                        # Nur updaten wenn vorher auch Fenster zu war
                        if dt_minutes > 0 and not self._previous_window_open:
                            error = self.model.update(
                                prev_temp=self._previous_temp,
                                current_temp=current_temp,
                                offset=self._previous_offset,
                                outside_temp=outside_temp,
                                window_open=False,
                                dt_minutes=dt_minutes,
                            )
                            
                            if self.model.rls.n_updates % 60 == 0:
                                logger.info(f"Model updated: {self.model}, error={error:.3f}°C")
                    
                    # MPC Optimierung
                    mpc_result = self.mpc.solve(
                        current_temp=current_temp,
                        target_temp=target_temp,
                        previous_offset=self._previous_offset,
                        outside_temps=np.full(self.mpc.config.horizon_steps, outside_temp),
                        window_states=np.zeros(self.mpc.config.horizon_steps),
                    )
                    
                    new_offset = mpc_result.optimal_offset
                    
                    logger.info(
                        f"MPC: T={current_temp:.1f}°C → target={target_temp}°C, "
                        f"offset={new_offset:.2f}°C, cost={mpc_result.cost_value:.2f}"
                    )
                    
                    # Controller-Log speichern
                    self.db.log_controller_step(
                        controller_type='mpc',
                        predicted_temps=mpc_result.predicted_temps.tolist(),
                        optimal_offsets=mpc_result.optimal_offsets.tolist(),
                        cost_value=mpc_result.cost_value,
                        solve_time_ms=mpc_result.solve_time_ms,
                    )
                
                # Offset anwenden
                await self.ha_client.set_temperature_offset(new_offset)
                
                # Erweiterte Messung für KI-Training speichern
                measurement = Measurement(
                    timestamp=loop_start,
                    room_temp=current_temp,
                    outside_temp=outside_temp,
                    window_open=window_open,
                    heating_active=sensor_data.heating_active,
                    control_offset=new_offset,
                    target_temp=target_temp,
                    mode=self._mode,
                )
                self.db.insert_measurement(measurement)
                
                # Erweitertes Logging für KI-Training
                self.db.log_training_sample(
                    timestamp=loop_start,
                    state={
                        'room_temp': current_temp,
                        'outside_temp': outside_temp,
                        'target_temp': target_temp,
                        'window_open': window_open,
                        'previous_temp': self._previous_temp,
                        'previous_offset': self._previous_offset,
                        'heating_active': sensor_data.heating_active,
                    },
                    action={'offset': new_offset},
                    reward=self._calculate_reward(current_temp, target_temp, new_offset, window_open),
                    model_params=self.model.params.to_dict(),
                )
                
                # State updaten
                self._previous_temp = current_temp
                self._previous_time = loop_start
                self._previous_offset = new_offset
                self._previous_window_open = window_open
                
                # Periodisch Modell speichern
                if self.model.rls.n_updates % 100 == 0:
                    self.model.save("data/model.json")
                
                # Warten bis zum nächsten Sample
                elapsed = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, sample_time - elapsed)
                await asyncio.sleep(sleep_time)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in control loop: {e}", exc_info=True)
                await asyncio.sleep(sample_time)
    
    async def run_experiment(self, experiment_type: str):
        """Führt ein Experiment durch."""
        logger.info(f"Running experiment: {experiment_type}")
        
        await self.ha_client.connect()
        
        try:
            if experiment_type == "step":
                result = await self.experiment_runner.run_step_response(interactive=True)
                
                # Modell mit identifizierten Parametern aktualisieren
                if result and result.identified_params:
                    logger.info(f"Identified parameters: {result.identified_params}")
                    self.model.params = result.identified_params
                    self.model.save("data/model.json")
            
            elif experiment_type == "prbs":
                result = await self.experiment_runner.run_prbs(interactive=True)
                if result:
                    logger.info(f"PRBS completed: {result.metrics}")
            
            elif experiment_type == "relay":
                # Hole aktuelle Solltemperatur aus HA
                await self.ha_client.connect()
                sensors = await self.ha_client.read_all_sensors()
                target_temp = sensors.thermostat_setpoint
                await self.ha_client.disconnect()
                
                result = await self.experiment_runner.run_relay_feedback(
                    target_temp=target_temp,
                    interactive=True
                )
                logger.info(f"Relay-feedback result: {result}")
            
            else:
                logger.error(f"Unknown experiment type: {experiment_type}")
        
        finally:
            await self.ha_client.disconnect()
    
    async def test_connection(self):
        """Testet Verbindung zu Home Assistant."""
        logger.info("Testing connection to Home Assistant...")
        
        try:
            await self.ha_client.connect()
            
            sensors = await self.ha_client.read_all_sensors()
            logger.info(f"Connection successful!")
            logger.info(f"Current readings: {sensors}")
            
            # Entity-Status prüfen
            for name, entity_id in self.config['entities'].items():
                if entity_id:
                    state = await self.ha_client.get_state(entity_id)
                    if state:
                        logger.info(f"  {name}: {entity_id} = {state.get('state')}")
                    else:
                        logger.warning(f"  {name}: {entity_id} NOT FOUND")
            
            await self.ha_client.disconnect()
            return True
        
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


async def main():
    """Hauptfunktion."""
    parser = argparse.ArgumentParser(
        description="Radiator Control System - MPC-basierte Heizungsregelung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py --test              # Verbindung testen
  python main.py --info              # Experiment-Übersicht anzeigen
  python main.py --stats             # Regelungs-Statistiken anzeigen
  python main.py --stats --days 14   # Statistiken der letzten 14 Tage
  python main.py --experiments       # Experiment-Ergebnisse anzeigen
  python main.py --experiment step   # Sprungantwort durchführen
  python main.py                     # Regelung starten
"""
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Pfad zur Konfigurationsdatei',
    )
    parser.add_argument(
        '--experiment', '-e',
        choices=['step', 'prbs', 'relay'],
        help='Führt ein Experiment zur Systemidentifikation durch',
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Testet die Verbindung zu Home Assistant',
    )
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Zeigt Übersicht der Experimente an',
    )
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Zeigt Regelungs-Statistiken (RMSE, Komfort-Quote, Trend)',
    )
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=7,
        help='Anzahl Tage für Statistiken (Standard: 7)',
    )
    parser.add_argument(
        '--experiments',
        action='store_true',
        help='Zeigt alle Experiment-Ergebnisse an',
    )
    
    args = parser.parse_args()
    
    controller = RadiatorController(config_path=args.config)
    
    # Info-Modus: nur Übersicht anzeigen
    if args.info:
        controller.experiment_runner.show_experiment_overview()
        return
    
    # Statistik-Modus
    if args.stats:
        controller.db.print_statistics_report(days=args.days)
        return
    
    # Experiment-Ergebnisse anzeigen
    if args.experiments:
        controller.db.print_experiment_report()
        return
    
    # Signal Handler für sauberes Beenden
    def signal_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(controller.stop())
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        if args.test:
            await controller.test_connection()
        elif args.experiment:
            await controller.run_experiment(args.experiment)
        else:
            await controller.start()
    except KeyboardInterrupt:
        pass
    finally:
        if controller._running:
            await controller.stop()


if __name__ == "__main__":
    asyncio.run(main())
