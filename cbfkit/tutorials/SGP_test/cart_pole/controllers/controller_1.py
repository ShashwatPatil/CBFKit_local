import jax.numpy as jnp
from typing import *
from jax import jit, Array, lax
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns


def controller_1(
    k_1: float,
    k_2: float,
    k_3: float,
    k_4: float,
    G: float,
    l: float,
) -> ControllerCallable:
    """
    Create a controller for the given dynamics.

    Args:
        #! USER-POPULATE

    Returns:
        controller (Callable): handle to function computing control

    """

    @jit
    def controller(t: float, x: Array) -> ControllerCallableReturns:
        """Computes control input (1x1).

        Args:
            t (float): time in sec
            x (Array): state vector (or estimate if using observer/estimator)

        Returns:
            unom (Array): 1x1 vector
            data: (dict): empty dictionary
        """
        # logging data
        u_nom = (
            (
                (x[3] / l) * jnp.cos(x[2]) * k_4
                + k_3 * (x[1] + k_1 * x[3] * jnp.cos(x[2]) + k_2 * jnp.sin(x[2]))
                + k_1 * jnp.sin(x[2]) * ((G / l) * jnp.cos(x[2]) - x[3] ** 2)
                + k_2 * x[3] * jnp.cos(x[2])
            )
        ) / ((k_1 / l) * jnp.cos(x[2]) ** 2 - 1)
        data = {"u_nom": u_nom}

        return jnp.array(u_nom), data

    return controller
