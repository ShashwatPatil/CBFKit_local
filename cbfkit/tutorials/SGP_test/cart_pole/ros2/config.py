
# Configuration settings for Ros2 application
# Defaults for cart_pole

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARN, ERROR, FATAL
QUEUE_SIZE = 10
MODULE_NAME = "cart_pole"

# Controller
CONTROLLER_NAME = "controller_1"
TIMER_INTERVAL = 0.1
CONTROLLER_PARAMS = {'k_1': 1.0, 'k_2': 1.0, 'k_3': 1.0, 'k_4': 1.0, 'G': 9.81, 'l': 0.5}

# Estimator
ESTIMATOR_NAME = "naive"

# Plant
PLANT_NAME = "plant"
PLANT_PARAMS = {'I': 0.099, 'M': 0.1, 'm': 0.1, 'G': 9.81, 'l': 0.5}
INTEGRATOR_NAME = "forward_euler"
DT = 0.01

# Sensor
SENSOR_NAME = "perfect"
SENSOR_PARAMS = {}
