
# Configuration settings for Ros2 application
# Defaults for cart_pole

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARN, ERROR, FATAL
QUEUE_SIZE = 10
MODULE_NAME = "cart_pole"

# Controller
CONTROLLER_NAME = "controller_1"
TIMER_INTERVAL = 0.1
CONTROLLER_PARAMS = {'k_1 : float': 1.0, 'k_2 : float': 1.0, 'k_3 : float': 1.0, 'k_4 : float': 1.0, 'G : float': 9.81, 'l : float': 0.5}

# Estimator
ESTIMATOR_NAME = "naive"

# Plant
PLANT_NAME = "plant"
PLANT_PARAMS = {'I : float': 0.099, 'M : float': 0.1, 'm : float': 0.1, 'G : float': 9.81, 'l : float': 0.5}
INTEGRATOR_NAME = "forward_euler"
DT = 0.01

# Sensor
SENSOR_NAME = "perfect"
SENSOR_PARAMS = {}
