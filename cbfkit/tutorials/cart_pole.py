from jax import Array, jit
import jax.numpy as jnp

from cbfkit.codegen.create_new_system import generate_model

# f + g * u  for cart pole
drift_dynamics = "[x[1], m * l * ( m * l * G * sin(x[2]) * cos(x[2]) + (I + m * l ** 2) * x[3] ** 2 * sin(x[1])) / ((I + m * l ** 2) * (M + m) - m ** 2 * l ** 2 * cos(x[2]) ** 2), x[3], -m * l * ((m + M) * G * sin(x[2]) + m * l * x[3] ** 2 * sin(x[2]) * cos(x[2])) / ((I + m * l ** 2) * (M + m) - m ** 2 * l ** 2 * cos(x[2]) ** 2)]"     # the f matrix
control_matrix = "[[0], (I + m * l ** 2) / ((I + m * l ** 2) * (M + m) - m ** 2 * l ** 2 * cos(x[2]) ** 2 ), [0], (-m * l * cos(x[2])) / ((I + m * l ** 2) * (M + m) - m ** 2 * l ** 2 * cos(x[2]) ** 2)]"          # the control matrix g

target_directory = "./SGP_test"
madel_name = "cart_pole"

params = {"dynamics": {"I : float": 0.099,"M : float": 2,"m : float": 0.2,"G : float": 9.81,"l : float": 0.5,}}

generate_model.generate_model(
    directory=target_directory,
    model_name=madel_name,
    drift_dynamics=drift_dynamics,
    control_matrix=control_matrix,
    params=params,
)


