import jax.numpy as jnp
from jax import jit, Array, lax
from typing import Optional, Union, Callable
from cbfkit.utils.user_types import DynamicsCallable, DynamicsCallableReturns
from .constants import *


def plant(I: float, M: float, m: float, G: float, l: float, **kwargs) -> DynamicsCallable:
    """
    Returns a function that represents the plant model,
    which computes the drift vector 'f' and control matrix 'g' based on the given state.

    States are the following:
        #! MANUALLY POPULATE

    Control inputs are the following:
        #! MANUALLY POPULATE

    Args:
        perturbation (Optional, Array): additive perturbation to the xdot dynamics
        kwargs: keyword arguments

    Returns:
        dynamics (Callable): takes state as input and returns dynamics components
            f, g of the form dx/dt = f(x) + g(x)u

    """

    @jit
    def dynamics(x: Array) -> DynamicsCallableReturns:
        """
        Computes the drift vector 'f' and control matrix 'g' based on the given state x.

        Args:
            x (Array): state vector

        Returns:
            dynamics (DynamicsCallable): takes state as input and returns dynamics components f, g
        """
        f = jnp.array(
            [
                x[1],
                (
                    m
                    * l
                    * (
                        m * l * G * jnp.sin(x[2]) * jnp.cos(x[2])
                        + (I + m * l**2) * x[3] ** 2 * jnp.sin(x[1])
                    )
                )
                / ((I + m * l**2) * (M + m) - m**2 * l**2 * jnp.cos(x[2]) ** 2),
                x[3],
                (
                    -m
                    * l
                    * (
                        (m + M) * G * jnp.sin(x[2])
                        + m * l * x[3] ** 2 * jnp.sin(x[2]) * jnp.cos(x[2])
                    )
                )
                / ((I + m * l**2) * (M + m) - m**2 * l**2 * jnp.cos(x[2]) ** 2),
            ]
        )
        g = jnp.array(
            [
                [0],
                (I + m * l**2)
                / ((I + m * l**2) * (M + m) - m**2 * l**2 * jnp.cos(x[2]) ** 2),
                [0],
                (-m * l * jnp.cos(x[2]))
                / ((I + m * l**2) * (M + m) - m**2 * l**2 * jnp.cos(x[2]) ** 2),
            ]
        )

        return f, g

    return dynamics
