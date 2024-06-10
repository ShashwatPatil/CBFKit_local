
# Configuration settings for Ros2 application
# Defaults for cart_pole

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARN, ERROR, FATAL
QUEUE_SIZE = 10
MODULE_NAME = "cart_pole"

# Controller
CONTROLLER_NAME = "controller_1"
TIMER_INTERVAL = 0.1
CONTROLLER_PARAMS = k_1 : float, k_2 : float, k_3 : float, k_4 : float, G : float, l : float, 

# Estimator
ESTIMATOR_NAME = "naive"

# Plant
PLANT_NAME = "plant"
PLANT_PARAMS = I : float, M : float, m : float, G : float, l : float, 
INTEGRATOR_NAME = "forward_euler"
DT = 0.01

# Sensor
SENSOR_NAME = "perfect"
SENSOR_PARAMS = {}
