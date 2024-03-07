"""
Utils funciton of the model module
"""
import jax
import jax.numpy as jnp


def check_symmetric(a: jax.Array, tol=1e-8):
    """
    Check if a matrix is symmetric

    :param a: (jax.Array): Matrix to check
    :param tol: (float): Tolerance for the check
    :return: (bool): True if the matrix is symmetric
    """
    return jnp.all(jnp.abs(a - a.T) < tol)
