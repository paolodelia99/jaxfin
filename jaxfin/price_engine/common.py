"""
Common functions for the price_engine submodules
"""

import jax
import jax.numpy as jnp
from .math import d1, cum_normal


def _compute_d1_d2(spots, strikes, expires, vols, discount_rates):
    """
    Compute the d1 and d2 terms in the Black-Scholes formula

    :param spots: Current spot price of the underlying
    :param strikes: Strike price of the option
    :param expires: Time to expiration of the option
    :param vols: Volatility of the underlying
    :param discount_rates: Risk-free rate
    :return: d1 and d2 terms
    """
    vol_sqrt_t = vols * jnp.sqrt(expires)

    _d1 = d1(spots, strikes, vols, expires, discount_rates)

    return [_d1, _d1 - vol_sqrt_t]


@jax.jit
def compute_undiscounted_call_prices(spots, strikes, expires, vols, discount_rates):
    """
    Compute the undiscounted call option prices

    :param spots: Current spot price of the underlying
    :param strikes: Strike price of the option
    :param expires: Time to expiration of the option
    :param vols: Volatility of the underlying
    :param discount_rates: Risk-free rate
    :return: Undiscounted call option prices
    """
    [_d1, _d2] = _compute_d1_d2(spots, strikes, expires, vols, discount_rates)

    return cum_normal(_d1) * spots - cum_normal(_d2) * strikes


@jax.jit
def compute_discounted_call_prices(spots, strikes, expires, vols, discount_rates):
    """
    Compute the discounted call option prices

    :param spots: Current spot price of the underlying
    :param strikes: Strike price of the option
    :param expires: Time to expiration of the option
    :param vols: Volatility of the underlying
    :param discount_rates: Risk-free rate
    :return: Discounted call option prices
    """
    [_d1, _d2] = _compute_d1_d2(spots, strikes, expires, vols, discount_rates)

    return cum_normal(_d1) * spots - cum_normal(_d2) * strikes * jnp.exp(
        (-discount_rates) * expires
    )
