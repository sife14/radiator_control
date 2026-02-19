"""Constants for the Radiator Control integration."""

DOMAIN = "radiator_control"
MANUFACTURER = "Radiator Control MPC"

# Config keys
CONF_THERMOSTAT_ENTITY = "thermostat_entity"
CONF_TEMP_SENSOR_ENTITY = "temp_sensor_entity"
CONF_WINDOW_SENSOR_ENTITY = "window_sensor_entity"
CONF_OUTSIDE_TEMP_ENTITY = "outside_temp_entity"
CONF_TEMP_CALIBRATION_ENTITY = "temp_calibration_entity"

# Options keys
CONF_WINDOW_ACTION = "window_action"
CONF_WINDOW_OFF_DELAY = "window_off_delay_seconds"

CONF_MPC_HORIZON = "mpc_horizon_minutes"
CONF_MPC_CONTROL_HORIZON = "mpc_control_horizon"
CONF_MPC_WEIGHT_COMFORT = "mpc_weight_comfort"
CONF_MPC_WEIGHT_ENERGY = "mpc_weight_energy"
CONF_MPC_WEIGHT_SMOOTHNESS = "mpc_weight_smoothness"

CONF_OFFSET_MIN = "offset_min"
CONF_OFFSET_MAX = "offset_max"
CONF_SAMPLE_TIME = "sample_time_seconds"

CONF_MODEL_INITIAL_TAU = "model_initial_tau"
CONF_MODEL_INITIAL_K_HEATER = "model_initial_k_heater"
CONF_MODEL_FORGETTING_FACTOR = "model_forgetting_factor"

# Window actions
WINDOW_ACTION_TURN_OFF = "turn_off"
WINDOW_ACTION_OFFSET = "offset"

# Defaults
DEFAULT_WINDOW_ACTION = WINDOW_ACTION_TURN_OFF
DEFAULT_WINDOW_OFF_DELAY = 30

DEFAULT_MPC_HORIZON = 240
DEFAULT_MPC_CONTROL_HORIZON = 15
DEFAULT_MPC_WEIGHT_COMFORT = 1.0
DEFAULT_MPC_WEIGHT_ENERGY = 0.1
DEFAULT_MPC_WEIGHT_SMOOTHNESS = 0.05

DEFAULT_OFFSET_MIN = -5.0
DEFAULT_OFFSET_MAX = 5.0
DEFAULT_SAMPLE_TIME = 60

DEFAULT_MODEL_INITIAL_TAU = 120.0
DEFAULT_MODEL_INITIAL_K_HEATER = 0.5
DEFAULT_MODEL_FORGETTING_FACTOR = 0.98

# Platforms
PLATFORMS = ["climate", "sensor", "number", "switch", "button"]

# Coordinator
SCAN_INTERVAL_SECONDS = 60

# Services
SERVICE_START_EXPERIMENT = "start_experiment"
SERVICE_STOP_EXPERIMENT = "stop_experiment"
SERVICE_RESET_MODEL = "reset_model"

# Attributes
ATTR_EXPERIMENT_TYPE = "experiment_type"
