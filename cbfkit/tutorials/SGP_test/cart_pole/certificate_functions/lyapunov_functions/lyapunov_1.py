"""
#! MANUALLY POPULATE (docstring)
"""
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev, Array, lax
from typing import List, Callable
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
)

N = 4


###############################################################################
# CLF
###############################################################################


def clf(
    k_1: float, k_2: float, k_3: float, G: float, l: float, **kwargs
) -> Callable[[Array], Array]:
    """Super-level set convention.

    Args:
        #! kwargs -- optional to manually populate

    Returns:
        ret (float): value of goal function evaluated at time and state

    """

    @jit
    def func(state_and_time: Array) -> Array:
        """Function to be evaluated.

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: clf value
        """
        x = state_and_time
        return (
            ((k_1 * jnp.cos(x[2]) ** 2 / l) - 1) * x[3] ** 2 / 2
            + (G / l) * (1 - jnp.cos(x[2]))
            + (k_3 / 2) * (x[1] + k_1 * x[3] * jnp.cos(x[2]) + k_2 * jnp.sin(x[2])) ** 2
        )

    return func


def clf_grad(
    k_1: float, k_2: float, k_3: float, G: float, l: float, **kwargs
) -> Callable[[Array], Array]:
    """Jacobian for the goal function defined by clf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    jacobian = jacfwd(clf(k_1, k_2, k_3, G, l, **kwargs))

    @jit
    def func(state_and_time: Array) -> Array:
        """_summary_

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: clf jacobian (gradient)
        """

        return jacobian(state_and_time)

    return func


def clf_hess(
    k_1: float, k_2: float, k_3: float, G: float, l: float, **kwargs
) -> Callable[[Array], Array]:
    """Hessian for the goal function defined by clf.

    Args:
        #! kwargs -- manually populate

    Returns:
        ret (float): value of goal function evaluated at time and state

    """
    hessian = jacrev(jacfwd(clf(k_1, k_2, k_3, G, l, **kwargs)))

    @jit
    def func(state_and_time: Array) -> Array:
        """_summary_

        Args:
            state_and_time (Array): concatenated state vector and time

        Returns:
            Array: clf hessian
        """

        return hessian(state_and_time)

    return func


###############################################################################
# CLF1
###############################################################################
clf1_package = certificate_package(clf, clf_grad, clf_hess, N)
