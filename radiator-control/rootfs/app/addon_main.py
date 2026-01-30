#!/usr/bin/env python3
"""
Radiator Control Add-on Main
============================
Startet Web-UI und Controller parallel.
"""

import asyncio
import json
import logging
import os
import signal
import threading
from datetime import datetime
from pathlib import Path

# Flask für Web-UI
from flask import Flask, render_template, jsonify, request
from waitress import serve

# Eigene Module
import sys
sys.path.insert(0, '/app')

from src.utils import load_config, setup_logging
from src.ha_client import HomeAssistantClient
from src.database import Database, Measurement
from src.model import ThermalModel, ThermalModelParams
from src.mpc_controller import MPCController, MPCConfig
from src.experiments import ExperimentRunner, ExperimentConfig
from src.climate_entity import ClimateEntity, create_climate_entity

# Logging
log_level = os.environ.get('LOG_LEVEL', 'info').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask App
app = Flask(__name__, 
            template_folder='/app/templates',
            static_folder='/app/static')

# Globale Referenzen
controller = None
experiment_runner = None
database = None
model = None
ha_client = None
climate_entity = None  # Climate-Entity für better-thermostat-ui-card
runtime_config = None  # Laufzeit-Konfiguration
current_status = {
    'running': False,
    'mode': 'idle',
    'room_temp': None,
    'target_temp': None,
    'outside_temp': None,
    'offset': 0,
    'window_open': False,
    'hvac_mode': None,
    'heating_active': False,
    'last_update': None,
    'experiment': None,
    'experiment_progress': 0,
    'entities': {},
}


# =============================================================================
# Web Routes
# =============================================================================

@app.route('/')
def index():
    """Hauptseite."""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Aktueller Status."""
    return jsonify(current_status)


@app.route('/api/stats')
def api_stats():
    """Statistiken."""
    if database is None:
        return jsonify({'error': 'Database not initialized'}), 500
    
    try:
        summary = database.get_performance_summary()
        daily = database.get_daily_statistics(days=7)
        
        return jsonify({
            'summary': summary,
            'daily': daily.to_dict('records') if len(daily) > 0 else [],
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiments')
def api_experiments():
    """Experiment-Ergebnisse."""
    if database is None:
        return jsonify({'error': 'Database not initialized'}), 500
    
    try:
        df = database.get_experiment_results()
        return jsonify({
            'experiments': df.to_dict('records') if len(df) > 0 else [],
        })
    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model')
def api_model():
    """Modellparameter."""
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    return jsonify({
        'params': model.params.to_dict(),
        'rmse': model.get_rmse(),
        'n_updates': model.rls.n_updates,
    })


@app.route('/api/control/start', methods=['POST'])
def api_start_control():
    """Startet die Regelung."""
    global current_status
    
    if current_status['running']:
        return jsonify({'error': 'Already running'}), 400
    
    current_status['running'] = True
    current_status['mode'] = 'control'
    
    # Starte Controller-Loop in separatem Thread
    threading.Thread(target=run_controller_async, daemon=True).start()
    
    return jsonify({'status': 'started'})


@app.route('/api/control/stop', methods=['POST'])
def api_stop_control():
    """Stoppt die Regelung."""
    global current_status
    
    current_status['running'] = False
    current_status['mode'] = 'stopping'
    
    return jsonify({'status': 'stopping'})


@app.route('/api/experiment/start', methods=['POST'])
def api_start_experiment():
    """Startet ein Experiment."""
    global current_status
    
    if current_status['running']:
        return jsonify({'error': 'Controller is running, stop it first'}), 400
    
    exp_type = request.json.get('type', 'step')
    
    if exp_type not in ['step', 'prbs', 'relay']:
        return jsonify({'error': f'Unknown experiment type: {exp_type}'}), 400
    
    current_status['mode'] = 'experiment'
    current_status['experiment'] = exp_type
    current_status['experiment_progress'] = 0
    
    # Starte Experiment in separatem Thread
    threading.Thread(target=run_experiment_async, args=(exp_type,), daemon=True).start()
    
    return jsonify({'status': 'started', 'type': exp_type})


@app.route('/api/experiment/stop', methods=['POST'])
def api_stop_experiment():
    """Stoppt laufendes Experiment."""
    global current_status
    
    if experiment_runner:
        experiment_runner.stop()
    
    current_status['mode'] = 'idle'
    current_status['experiment'] = None
    
    return jsonify({'status': 'stopped'})


@app.route('/api/thermostat/on', methods=['POST'])
def api_thermostat_on():
    """Schaltet Thermostat ein."""
    def async_on():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(ha_client.connect())
            loop.run_until_complete(ha_client.turn_on())
            loop.run_until_complete(ha_client.disconnect())
        except Exception as e:
            logger.error(f"Error turning on thermostat: {e}")
    
    threading.Thread(target=async_on, daemon=True).start()
    current_status['hvac_mode'] = 'heat'
    return jsonify({'status': 'on'})


@app.route('/api/thermostat/off', methods=['POST'])
def api_thermostat_off():
    """Schaltet Thermostat aus."""
    def async_off():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(ha_client.connect())
            loop.run_until_complete(ha_client.turn_off())
            loop.run_until_complete(ha_client.disconnect())
        except Exception as e:
            logger.error(f"Error turning off thermostat: {e}")
    
    threading.Thread(target=async_off, daemon=True).start()
    current_status['hvac_mode'] = 'off'
    return jsonify({'status': 'off'})


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get/Set Konfiguration."""
    config_path = '/app/config.yaml'
    runtime_config_path = os.environ.get('DATA_PATH', '/data/radiator_control') + '/runtime_config.json'
    
    if request.method == 'GET':
        config = load_config(config_path)
        
        # Runtime overrides laden
        try:
            import json
            if os.path.exists(runtime_config_path):
                with open(runtime_config_path, 'r') as f:
                    runtime_cfg = json.load(f)
                # Merge
                for key in runtime_cfg:
                    if key in config:
                        config[key].update(runtime_cfg[key])
        except Exception as e:
            logger.warning(f"Could not load runtime config: {e}")
        
        return jsonify({
            'mpc': {
                'horizon_minutes': config.get('mpc', {}).get('horizon_minutes', 120),
                'weight_comfort': config.get('mpc', {}).get('weight_comfort', 1.0),
                'weight_energy': config.get('mpc', {}).get('weight_energy', 0.1),
                'weight_smoothness': config.get('mpc', {}).get('weight_smoothness', 0.05),
            },
            'control': {
                'offset_min': config.get('control', {}).get('offset_min', -5),
                'offset_max': config.get('control', {}).get('offset_max', 5),
                'window_action': config.get('control', {}).get('window_action', 'turn_off'),
                'window_off_delay': config.get('control', {}).get('window_off_delay_seconds', 30),
            },
            'model': {
                'forgetting_factor': config.get('model', {}).get('rls_forgetting_factor', 0.98),
            },
            'entities': config.get('entities', {}),
        })
    
    elif request.method == 'POST':
        try:
            import json
            new_settings = request.json
            
            # Speichere als Runtime-Config (wird beim nächsten Loop verwendet)
            runtime_cfg = {}
            if 'mpc' in new_settings:
                runtime_cfg['mpc'] = new_settings['mpc']
            if 'control' in new_settings:
                runtime_cfg['control'] = new_settings['control']
            if 'model' in new_settings:
                runtime_cfg['model'] = {
                    'rls_forgetting_factor': new_settings['model'].get('forgetting_factor', 0.98),
                }
            
            # Speichern
            os.makedirs(os.path.dirname(runtime_config_path), exist_ok=True)
            with open(runtime_config_path, 'w') as f:
                json.dump(runtime_cfg, f, indent=2)
            
            return jsonify({'status': 'saved'})
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return jsonify({'error': str(e)}), 500


@app.route('/api/model/reset', methods=['POST'])
def api_model_reset():
    """Setzt Modell zurück."""
    global model
    
    config = load_config('/app/config.yaml')
    model_cfg = config.get('model', {})
    
    model = ThermalModel(
        initial_params=ThermalModelParams(
            tau=model_cfg.get('initial_tau', 120),
            k_heater=model_cfg.get('initial_k_heater', 0.5),
        ),
        forgetting_factor=model_cfg.get('rls_forgetting_factor', 0.98),
    )
    
    # Speichern
    data_path = os.environ.get('DATA_PATH', '/data/radiator_control')
    model.save(f"{data_path}/model.json")
    
    logger.info("Model reset to initial values")
    return jsonify({'status': 'reset'})


# =============================================================================
# Controller Logic
# =============================================================================

def run_controller_async():
    """Führt Controller-Loop in neuem Event-Loop aus."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(controller_loop())


async def controller_loop():
    """Hauptregelschleife."""
    global current_status, model
    
    config = load_config('/app/config.yaml')
    sample_time = config['control'].get('sample_time', 60)
    
    # Lade Runtime-Config falls vorhanden
    runtime_config_path = os.environ.get('DATA_PATH', '/data/radiator_control') + '/runtime_config.json'
    try:
        import json
        if os.path.exists(runtime_config_path):
            with open(runtime_config_path, 'r') as f:
                runtime_cfg = json.load(f)
                for key in runtime_cfg:
                    if key in config:
                        config[key].update(runtime_cfg[key])
    except Exception:
        pass
    
    # Window action config
    window_action = config.get('control', {}).get('window_action', 'turn_off')
    window_off_delay = config.get('control', {}).get('window_off_delay_seconds', 30)
    
    try:
        await ha_client.connect()
        
        previous_temp = None
        previous_time = None
        previous_offset = 0.0
        previous_window = False
        saved_hvac_mode = None  # Für window_action=turn_off
        window_open_since = None  # Für window_off_delay
        
        while current_status['running']:
            loop_start = datetime.now()
            
            # Sensoren lesen
            sensor_data = await ha_client.read_all_sensors()
            current_temp = sensor_data.room_temp
            outside_temp = sensor_data.outside_temp or 5.0
            window_open = sensor_data.window_open
            target_temp = sensor_data.thermostat_setpoint
            hvac_mode = sensor_data.hvac_mode
            
            # Status aktualisieren
            current_status['room_temp'] = current_temp
            current_status['target_temp'] = target_temp
            current_status['outside_temp'] = outside_temp
            current_status['window_open'] = window_open
            current_status['hvac_mode'] = hvac_mode
            current_status['heating_active'] = sensor_data.heating_active
            current_status['last_update'] = loop_start.isoformat()
            current_status['entities'] = config.get('entities', {})
            
            if window_open:
                current_status['mode'] = 'window_open'
                
                if window_action == 'turn_off':
                    # Thermostat ausschalten bei offenem Fenster
                    if window_open_since is None:
                        window_open_since = loop_start
                    
                    # Delay abwarten
                    elapsed = (loop_start - window_open_since).total_seconds()
                    if elapsed >= window_off_delay:
                        if hvac_mode != 'off':
                            if saved_hvac_mode is None:
                                saved_hvac_mode = hvac_mode or 'heat'
                            await ha_client.turn_off()
                            logger.info(f"Window open for {elapsed:.0f}s - turned thermostat OFF")
                        new_offset = 0  # Offset reset
                    else:
                        # Noch im Delay - nur Offset maximieren
                        new_offset = config['control'].get('offset_max', 5.0)
                else:
                    # window_action == 'offset': nur Offset maximieren
                    new_offset = config['control'].get('offset_max', 5.0)
            else:
                # Fenster geschlossen
                window_open_since = None
                current_status['mode'] = 'control'
                
                # Thermostat wieder einschalten falls es ausgeschaltet wurde
                if saved_hvac_mode is not None and hvac_mode == 'off':
                    await ha_client.set_hvac_mode(saved_hvac_mode)
                    logger.info(f"Window closed - restored thermostat to {saved_hvac_mode}")
                    saved_hvac_mode = None
                
                # Modell updaten
                if previous_temp is not None and previous_time is not None:
                    dt_minutes = (loop_start - previous_time).total_seconds() / 60
                    if dt_minutes > 0 and not previous_window:
                        model.update(
                            prev_temp=previous_temp,
                            current_temp=current_temp,
                            offset=previous_offset,
                            outside_temp=outside_temp,
                            window_open=False,
                            dt_minutes=dt_minutes,
                        )
                
                # MPC Optimierung
                mpc = MPCController(
                    model=model,
                    config=MPCConfig(
                        horizon_steps=config['mpc'].get('horizon_minutes', 120) // 5,
                        control_horizon=10,
                        dt_minutes=5.0,
                        weight_comfort=config['mpc'].get('weight_comfort', 1.0),
                        weight_energy=config['mpc'].get('weight_energy', 0.1),
                        weight_smoothness=config['mpc'].get('weight_smoothness', 0.05),
                        offset_min=config['control'].get('offset_min', -5.0),
                        offset_max=config['control'].get('offset_max', 5.0),
                    ),
                )
                
                import numpy as np
                result = mpc.solve(
                    current_temp=current_temp,
                    target_temp=target_temp,
                    previous_offset=previous_offset,
                    outside_temps=np.full(mpc.config.horizon_steps, outside_temp),
                    window_states=np.zeros(mpc.config.horizon_steps),
                )
                new_offset = result.optimal_offset
            
            # Offset anwenden
            await ha_client.set_temperature_offset(new_offset)
            current_status['offset'] = new_offset
            
            # Messung speichern
            measurement = Measurement(
                timestamp=loop_start,
                room_temp=current_temp,
                outside_temp=outside_temp,
                window_open=window_open,
                heating_active=sensor_data.heating_active,
                control_offset=new_offset,
                target_temp=target_temp,
                mode=current_status['mode'],
            )
            database.insert_measurement(measurement)
            
            # State updaten
            previous_temp = current_temp
            previous_time = loop_start
            previous_offset = new_offset
            previous_window = window_open
            
            # Modell periodisch speichern
            if model.rls.n_updates % 100 == 0:
                data_path = os.environ.get('DATA_PATH', '/data/radiator_control')
                model.save(f"{data_path}/model.json")
            
            # Warten
            elapsed = (datetime.now() - loop_start).total_seconds()
            await asyncio.sleep(max(0, sample_time - elapsed))
        
    except Exception as e:
        logger.error(f"Controller error: {e}", exc_info=True)
    finally:
        await ha_client.set_temperature_offset(0)
        await ha_client.disconnect()
        current_status['mode'] = 'idle'
        
        # Modell speichern
        data_path = os.environ.get('DATA_PATH', '/data/radiator_control')
        model.save(f"{data_path}/model.json")


def run_experiment_async(exp_type: str):
    """Führt Experiment in neuem Event-Loop aus."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_experiment(exp_type))


async def run_experiment(exp_type: str):
    """Führt ein Experiment durch."""
    global current_status, model
    
    try:
        await ha_client.connect()
        
        # Progress Callback
        async def progress_callback(progress: float, message: str):
            current_status['experiment_progress'] = progress
            logger.info(f"Experiment {progress*100:.0f}%: {message}")
        
        experiment_runner.set_progress_callback(progress_callback)
        
        if exp_type == 'step':
            result = await experiment_runner.run_step_response(interactive=False)
            if result and result.identified_params:
                model.params = result.identified_params
                data_path = os.environ.get('DATA_PATH', '/data/radiator_control')
                model.save(f"{data_path}/model.json")
                
        elif exp_type == 'prbs':
            result = await experiment_runner.run_prbs(interactive=False)
            
        elif exp_type == 'relay':
            sensors = await ha_client.read_all_sensors()
            result = await experiment_runner.run_relay_feedback(
                target_temp=sensors.thermostat_setpoint,
                interactive=False
            )
        
        current_status['experiment_progress'] = 1.0
        
    except asyncio.CancelledError:
        logger.info("Experiment cancelled")
    except Exception as e:
        logger.error(f"Experiment error: {e}", exc_info=True)
    finally:
        await ha_client.set_temperature_offset(0)
        await ha_client.disconnect()
        current_status['mode'] = 'idle'
        current_status['experiment'] = None


# =============================================================================
# HA Sensor Publishing
# =============================================================================

async def publish_sensors():
    """Veröffentlicht Sensoren in Home Assistant."""
    global climate_entity
    
    config = load_config('/app/config.yaml')
    create_climate = config.get('climate_entity', {}).get('create', False)
    climate_entity_id = config.get('climate_entity', {}).get('entity_id', 'climate.radiator_mpc_control')
    climate_friendly_name = config.get('climate_entity', {}).get('friendly_name', 'Radiator MPC Control')
    
    # Climate Entity erstellen falls gewünscht
    if create_climate and climate_entity is None:
        try:
            await ha_client.connect()
            climate_entity = await create_climate_entity(
                ha_client=ha_client,
                entity_id=climate_entity_id,
                friendly_name=climate_friendly_name,
            )
            logger.info(f"Created climate entity: {climate_entity_id}")
        except Exception as e:
            logger.error(f"Failed to create climate entity: {e}")
    
    while True:
        try:
            if ha_client and current_status['room_temp'] is not None:
                await ha_client.connect()
                
                rmse = model.get_rmse() if model else 0
                
                # RMSE Sensor
                await ha_client.set_state(
                    'sensor.radiator_control_rmse',
                    state=round(rmse, 3),
                    attributes={
                        'unit_of_measurement': '°C',
                        'friendly_name': 'Radiator Control RMSE',
                        'icon': 'mdi:chart-line',
                    }
                )
                
                # Offset Sensor
                await ha_client.set_state(
                    'sensor.radiator_control_offset',
                    state=round(current_status['offset'], 2),
                    attributes={
                        'unit_of_measurement': '°C',
                        'friendly_name': 'Radiator Control Offset',
                        'icon': 'mdi:thermometer-plus',
                    }
                )
                
                # Mode Sensor
                await ha_client.set_state(
                    'sensor.radiator_control_mode',
                    state=current_status['mode'],
                    attributes={
                        'friendly_name': 'Radiator Control Mode',
                        'icon': 'mdi:radiator',
                    }
                )
                
                # Climate Entity aktualisieren
                if climate_entity:
                    hvac_action = 'heating' if current_status.get('heating_active') else 'idle'
                    if current_status.get('hvac_mode') == 'off':
                        hvac_action = 'off'
                    
                    await climate_entity.update(
                        current_temp=current_status['room_temp'],
                        target_temp=current_status['target_temp'],
                        hvac_mode=current_status.get('hvac_mode', 'heat'),
                        hvac_action=hvac_action,
                        control_offset=current_status['offset'],
                        model_rmse=rmse,
                        window_open=current_status['window_open'],
                    )
                
                await ha_client.disconnect()
                
        except Exception as e:
            logger.error(f"Error publishing sensors: {e}")
        
        await asyncio.sleep(60)


# =============================================================================
# Main
# =============================================================================

def initialize():
    """Initialisiert alle Komponenten."""
    global database, model, ha_client, experiment_runner
    
    config = load_config('/app/config.yaml')
    
    # Database
    database = Database(config['database']['path'])
    
    # Model
    data_path = os.environ.get('DATA_PATH', '/data/radiator_control')
    model_path = Path(f"{data_path}/model.json")
    
    if model_path.exists():
        model = ThermalModel.load(str(model_path))
        logger.info("Loaded existing model")
    else:
        model_cfg = config.get('model', {})
        model = ThermalModel(
            initial_params=ThermalModelParams(
                tau=model_cfg.get('initial_tau', 120),
                k_heater=model_cfg.get('initial_k_heater', 0.5),
            ),
            forgetting_factor=model_cfg.get('rls_forgetting_factor', 0.98),
        )
        logger.info("Created new model")
    
    # HA Client
    ha_client = HomeAssistantClient(
        url=config['homeassistant']['url'],
        token=config['homeassistant']['token'],
        entities=config['entities'],
    )
    
    # Experiment Runner
    experiment_runner = ExperimentRunner(
        ha_client=ha_client,
        database=database,
    )
    
    logger.info("Initialization complete")


def main():
    """Hauptfunktion."""
    logger.info("Starting Radiator Control Add-on...")
    
    # Initialisierung
    initialize()
    
    # Sensor Publishing Thread
    def sensor_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(publish_sensors())
    
    threading.Thread(target=sensor_loop, daemon=True).start()
    
    # Web-UI starten (blockiert)
    logger.info("Starting Web UI on port 5000...")
    serve(app, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
