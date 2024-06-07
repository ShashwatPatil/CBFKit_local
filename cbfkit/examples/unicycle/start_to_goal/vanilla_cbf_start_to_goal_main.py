import jax.numpy as jnp

import cbfkit.systems.unicycle.models.olfatisaber2002approximate as unicycle
import cbfkit.simulation.simulator as sim
from cbfkit.controllers.model_based.cbf_clf_controllers import (
    vanilla_cbf_clf_qp_controller as cbf_controller,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    concatenate_certificates,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)

from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor
from cbfkit.estimators import naive as estimator

# Simulation parameters
tf = 10.0                # time frame
dt = 0.01
file_path = "examples/unicycle/start_to_goal/results/"

approx_unicycle_dynamics = unicycle.plant(l=1.0)
init_state = jnp.array([-3.5, -0.25, jnp.pi / 4])
desired_state = jnp.array([3.75, 3.5, -jnp.pi / 4])
actuation_constraints = jnp.array([100.0, 100.0])  # Effectively, no control limits

approx_uniycle_nom_controller = unicycle.controllers.proportional_controller(
    dynamics=approx_unicycle_dynamics,
    Kp_pos=0.8,
    Kp_theta=4,
    desired_state=desired_state,
)

# for this unicycle KP_pos = x then KP_theta  = 5 * x



obstacles = [
    (1.0, 2.0, 0.0),
    (-1.0, 1.0, 0.0),
    (0.5, -1.0, 0.0),
]
ellipsoids = [
    (0.5, 1.5),
    (1.0, 0.75),
    (0.75, 0.5),
]

barriers = [
    unicycle.certificate_functions.barrier_functions.obstacle_ca(
        certificate_conditions=zeroing_barriers.linear_class_k(0.5),
        obstacle=obs,
        ellipsoid=ell,
    )
    for obs, ell in zip(obstacles, ellipsoids)
]
barrier_packages = concatenate_certificates(*barriers)

controller = cbf_controller(
    control_limits=actuation_constraints,
    nominal_input=approx_uniycle_nom_controller,
    dynamics_func=approx_unicycle_dynamics,
    barriers=barrier_packages,
)

# Simulation imports

x, u, z, p, data, data_keys = sim.execute(
    x0=init_state,
    dt=dt,
    num_steps=int(tf / dt),
    dynamics=approx_unicycle_dynamics,
    integrator=integrator,
    controller=controller,
    sensor=sensor,
    estimator=estimator,
    filepath=file_path + "vanilla_cbf_results",
)


plot = 1
animate = 1
save = 0

if plot:
    from visualizations import plot_trajectory

    plot_trajectory(
        states=x,
        desired_state=desired_state,
        desired_state_radius=0.1,
        obstacles=obstacles,
        ellipsoids=ellipsoids,
        x_lim=(-2, 6),
        y_lim=(-2, 6),
        title="System Behavior",
    )

if animate:
    from visualizations import animate

    animate(
        states=x,
        estimates=z,
        desired_state=desired_state,
        desired_state_radius=0.1,
        x_lim=(-5, 5),
        y_lim=(-5, 5),
        dt=dt,
        title="System Behavior",
        obstacles=obstacles,
        ellipsoids=ellipsoids,
        save_animation=save,
        animation_filename=file_path + "vanilla_cbf_control.gif",
    )
