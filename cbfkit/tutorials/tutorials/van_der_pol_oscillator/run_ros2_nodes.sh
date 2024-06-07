#!/bin/bash

    # Source ROS2 environment
    source ~/ros2_humble/install/setup.bash

    # Run ROS2 node scripts
    cd ..
  python3 van_der_pol_oscillator/ros2/plant_model.py &
  python3 van_der_pol_oscillator/ros2/sensor.py &
  python3 van_der_pol_oscillator/ros2/estimator.py &
  python3 van_der_pol_oscillator/ros2/controller.py