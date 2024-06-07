import jax.numpy as jnp
from typing import *
from jax import jit, Array, lax
from cbfkit.utils.user_types import ControllerCallable, ControllerCallableReturns


def controller_1(
    k_p: float,
    epsilon: float,
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
        u_nom = x[0] * (1 - k_p) - epsilon * (1 - x[0] ** 2) * x[1]
        data = {"u_nom": u_nom}

        return jnp.array(u_nom), data

    return controller
