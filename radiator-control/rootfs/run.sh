#!/usr/bin/with-contenv bashio
# shellcheck shell=bash
# ==============================================================================
# Home Assistant Add-on: Radiator Control MPC
# Runs the MPC heating controller with Web UI
# ==============================================================================

# Konfiguration aus Add-on Options laden
export THERMOSTAT_ENTITY=$(bashio::config 'thermostat_entity')
export TEMP_SENSOR_ENTITY=$(bashio::config 'temp_sensor_entity')
export WINDOW_SENSOR_ENTITY=$(bashio::config 'window_sensor_entity')
export OUTSIDE_TEMP_ENTITY=$(bashio::config 'outside_temp_entity')
export TEMP_CALIBRATION_ENTITY=$(bashio::config 'temp_calibration_entity')

# Fenster-Verhalten
export WINDOW_ACTION=$(bashio::config 'window_action')
export WINDOW_OFF_DELAY=$(bashio::config 'window_off_delay_seconds')

# MPC Parameter
export MPC_HORIZON=$(bashio::config 'mpc_horizon_minutes')
export MPC_CONTROL_HORIZON=$(bashio::config 'mpc_control_horizon')
export MPC_WEIGHT_COMFORT=$(bashio::config 'mpc_weight_comfort')
export MPC_WEIGHT_ENERGY=$(bashio::config 'mpc_weight_energy')
export MPC_WEIGHT_SMOOTHNESS=$(bashio::config 'mpc_weight_smoothness')

# Control Parameter
export OFFSET_MIN=$(bashio::config 'offset_min')
export OFFSET_MAX=$(bashio::config 'offset_max')
export SAMPLE_TIME=$(bashio::config 'sample_time_seconds')

# Modell Parameter
export MODEL_INITIAL_TAU=$(bashio::config 'model_initial_tau')
export MODEL_INITIAL_K=$(bashio::config 'model_initial_k_heater')
export MODEL_FORGETTING=$(bashio::config 'model_forgetting_factor')

# Climate Entity
export CREATE_CLIMATE=$(bashio::config 'create_climate_entity')
export CLIMATE_NAME=$(bashio::config 'climate_entity_name')

export LOG_LEVEL=$(bashio::config 'log_level')

# Home Assistant API
export HA_TOKEN="${SUPERVISOR_TOKEN}"
export HA_URL="http://supervisor/core"

# Datenpfad
export DATA_PATH="/data/radiator_control"
mkdir -p "${DATA_PATH}"

bashio::log.info "Starting Radiator Control MPC v1.1.0..."
bashio::log.info "Thermostat: ${THERMOSTAT_ENTITY}"
bashio::log.info "Temp Sensor: ${TEMP_SENSOR_ENTITY}"
bashio::log.info "Window Action: ${WINDOW_ACTION}"
bashio::log.info "MPC Horizon: ${MPC_HORIZON} min"
bashio::log.info "Data Path: ${DATA_PATH}"

# Konfiguration schreiben
cat > /app/config.yaml << EOF
homeassistant:
  url: "${HA_URL}"
  token: "${HA_TOKEN}"

entities:
  thermostat: "${THERMOSTAT_ENTITY}"
  temp_sensor: "${TEMP_SENSOR_ENTITY}"
  window_sensor: "${WINDOW_SENSOR_ENTITY}"
  outside_temp: "${OUTSIDE_TEMP_ENTITY}"
  temp_calibration: "${TEMP_CALIBRATION_ENTITY}"

database:
  path: "${DATA_PATH}/measurements.db"

model:
  initial_tau: ${MODEL_INITIAL_TAU}
  initial_k_heater: ${MODEL_INITIAL_K}
  rls_forgetting_factor: ${MODEL_FORGETTING}

mpc:
  horizon_minutes: ${MPC_HORIZON}
  control_horizon: ${MPC_CONTROL_HORIZON}
  weight_comfort: ${MPC_WEIGHT_COMFORT}
  weight_energy: ${MPC_WEIGHT_ENERGY}
  weight_smoothness: ${MPC_WEIGHT_SMOOTHNESS}

control:
  sample_time: ${SAMPLE_TIME}
  offset_min: ${OFFSET_MIN}
  offset_max: ${OFFSET_MAX}
  window_action: "${WINDOW_ACTION}"
  window_off_delay_seconds: ${WINDOW_OFF_DELAY}

climate_entity:
  create: ${CREATE_CLIMATE}
  entity_id: "climate.radiator_mpc_control"
  friendly_name: "${CLIMATE_NAME}"

logging:
  level: ${LOG_LEVEL^^}
  file: "${DATA_PATH}/radiator_control.log"
EOF

# Web UI und Controller starten
cd /app
exec python3 /app/addon_main.py
