"""
Common math functions related to the Black Scholes model (univariate and multivariate) for the price_engine submodules
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit

from .norm import cum_normal


def _compute_d1_d2(spots, strikes, expires, vols, discount_rates):
    """
    Compute the d1 and d2 terms in the Black-Scholes formula.

    Args:
        spots (float): Current spot price of the underlying.
        strikes (float): Strike price of the option.
        expires (float): Time to expiration of the option.
        vols (float): Volatility of the underlying.
        discount_rates (float): Risk-free rate.

    Returns:
        tuple: A tuple containing the d1 and d2 terms.
    """
    vol_sqrt_t = vols * jnp.sqrt(expires)

    _d1 = d1_bs(spots, strikes, vols, expires, discount_rates)

    return [_d1, _d1 - vol_sqrt_t]


@jit
def d1_bs(spots, strikes, vols, expires, discount_rates):
    """
    Calculate the d1 term in the Black-Scholes formula.

    Args:
        spots (float): Current spot price of the underlying.
        strikes (float): Strike price of the option.
        vols (float): Volatility of the underlying.
        expires (float): Time to expiration of the option.
        discount_rates (float): Risk-free rate.

    Returns:
        float: The d1 term.
    """
    vol_sqrt_t = vols * jnp.sqrt(expires)

    return jnp.divide(
        (jnp.log(spots / strikes) + (discount_rates + (vols**2 / 2)) * expires),
        vol_sqrt_t,
    )


@jit
def compute_undiscounted_call_prices(spots, strikes, expires, vols, discount_rates):
    """
    Compute the undiscounted call option prices.

    Args:
        spots (float): Current spot price of the underlying.
        strikes (float): Strike price of the option.
        expires (float): Time to expiration of the option.
        vols (float): Volatility of the underlying.
        discount_rates (float): Risk-free rate.

    Returns:
        float: Undiscounted call option prices.
    """
    [_d1, _d2] = _compute_d1_d2(spots, strikes, expires, vols, discount_rates)

    return cum_normal(_d1) * spots - cum_normal(_d2) * strikes


@jit
def compute_discounted_call_prices(spots, strikes, expires, vols, discount_rates):
    """
    Compute the discounted call option prices.

    Args:
        spots (float): Current spot price of the underlying.
        strikes (float): Strike price of the option.
        expires (float): Time to expiration of the option.
        vols (float): Volatility of the underlying.
        discount_rates (float): Risk-free rate.

    Returns:
        float: Discounted call option prices.
    """
    [_d1, _d2] = _compute_d1_d2(spots, strikes, expires, vols, discount_rates)

    return cum_normal(_d1) * spots - cum_normal(_d2) * strikes * jnp.exp(
        (-discount_rates) * expires
    )


# Margrabe related functions


@jit
def d1_margrabe(spots_1, spots_2, sigma_sqrt_t) -> jax.Array:
    """
    Calculate the d1 term for the margrabe formula

    Args:
        spots_1 (_type_): Current price of the first underlying of the margrabe formula
        spots_2 (_type_): Current price of the second underlying of the margrabe formula
        sigma_sqrt_t (_type_): sigma_sqrt_t term

    Returns:
        _type_: The d1 margrabe term
    """
    return (jnp.log(spots_2 / spots_1) / sigma_sqrt_t) + 0.5 * sigma_sqrt_t


@jit
def d1_d2_margrabe(spots_1, spots_2, expires, sigma_1, sigma_2, corr) -> Tuple[jax.Array, jax.Array]:
    """
    Calculate the d1 and d2 term for the Margrabe formula

    Args:
        spots_1 (_type_): Spot_1 price
        spots_2 (_type_): Spot_2 price
        expires (_type_): Time to expiration of the option
        sigma_1 (_type_): Volatility of the first underlying
        sigma_2 (_type_): Volatility of the second underlying
        corr (_type_): Correlation term of the two underlyings

    Returns:
        Tuple[jax.Array, jax.Array]: d1 and d2
    """
    sigma = jnp.sqrt(
        jnp.sum(jnp.asarray([sigma_1, sigma_2]) ** 2)
        - 2 * corr * jnp.prod(jnp.asarray([sigma_1, sigma_2]))
    )

    sigma_sqrt_t = sigma * jnp.sqrt(expires)
    d1 = d1_margrabe(spots_1, spots_2, sigma_sqrt_t)
    d2 = d1 - sigma_sqrt_t

    return d1, d2
