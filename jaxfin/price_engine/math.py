"""
Common math functions used in multiple submodules in the price_engine module
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.stats.norm import pdf

_SQRT_2 = jnp.sqrt(2.0)


@jax.jit
def d1(spots, strikes, vols, expires, discount_rates):
    """
    Calculate the d1 term in the Black-Scholes formula

    :param spots: Current spot price of the underlying
    :param strikes: Strike price of the option
    :param vols: Volatility of the underlying
    :param expires: Time to expiration of the option
    :param discount_rates: Risk-free rate
    :return: d1 term
    """
    vol_sqrt_t = vols * jnp.sqrt(expires)

    return jnp.divide(
        (jnp.log(spots / strikes) + (discount_rates + (vols**2 / 2)) * expires),
        vol_sqrt_t,
    )


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
