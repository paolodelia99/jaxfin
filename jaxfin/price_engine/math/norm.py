"""
Common standard normal related functions used in multiple submodules in the price_engine module
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.stats.norm import pdf

_SQRT_2 = jnp.sqrt(2.0)


@jax.jit
def cum_normal(x):
    """
    Cumulative normal distribution function

    :param x: Input value
    :return: Cumulative normal distribution value
    """
    return (erf(x / _SQRT_2) + 1) / 2


@jax.jit
def density_normal(x):
    """
    Normal distribution function

    :param x: Input value
    :return: Normal distribution value
    """
    return pdf(x)
