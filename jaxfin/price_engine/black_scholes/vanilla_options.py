"""
Black Scholes prices for Vanilla European options
"""
from typing import Callable, Optional, Union, TypeVar

import jax
import jax.numpy as jnp
from jax import grad, jit

from ..math.bs_common import compute_discounted_call_prices
from ..utils.arrays import cast_arrays


F = TypeVar("F", bound=Callable)


def bs_price(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    are_calls: Optional[jax.Array] = None,
    dtype: Optional[jnp.dtype] = None,
) -> jax.Array:
    """
    Compute the option prices for european options using the Black-Scholes model.

    Args:
        spots (jax.Array): Array of current asset prices.
        strikes (jax.Array): Array of option strike prices.
        expires (jax.Array): Array of option expiration times.
        vols (jax.Array): Array of option volatility values.
        discount_rates (jax.Array): Array of risk-free interest rates. Defaults to None.
        dividend_rates (jax.Array, optional): Array of dividend rates. Defaults to None.
        are_calls (jax.Array): Array of booleans indicating whether options are calls (True) or puts (False).
        dtype (jnp.dtype, optional): Data type of the output. Defaults to None.

    Returns:
        jax.Array: Array of computed option prices.
    """
    [spots, strikes, expires, vols] = cast_arrays(
        [spots, strikes, expires, vols], dtype
    )

    discount_factors = jnp.exp(-discount_rates * expires)

    calls = compute_discounted_call_prices(
        spots, strikes, expires, vols, discount_rates
    )

    if are_calls is None:
        return calls

    puts = calls + (strikes * discount_factors) - spots
    return jnp.where(are_calls, calls, puts)


@jit
def delta_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Optional[jax.Array] = None,
    dtype: Optional[jnp.dtype] = None,
) -> jax.Array:
    """
    Calculate the delta of a call/put option under the BS model (scalar version)

    :param spots: (Union[jax.Array, float]): Array of current asset prices.
    :param strikes: (Union[jax.Array, float]): Array of option strike prices.
    :param expires: (Union[jax.Array, float]): Array of option expiration times.
    :param vols: (Union[jax.Array, float]): Array of option volatility values.
    :param discount_rates: (Union[jax.Array, float]): Array of risk-free interest rates.
    :param are_calls: (jax.Array): Array of booleans indicating whether options
                                   are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of delta of the given options.
    """
    return grad(bs_price, argnums=0)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


@jit
def gamma_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Optional[jax.Array] = None,
    dtype: Optional[jnp.dtype] = None,
) -> jax.Array:
    """
    Calculate the gamma of a european option under the BS model (scalar version)

    :param spots: (Union[jax.Array, float]): Array of current asset prices.
    :param strikes: (Union[jax.Array, float]): Array of option strike prices.
    :param expires: (Union[jax.Array, float]): Array of option expiration times.
    :param vols: (Union[jax.Array, float]): Array of option volatility values.
    :param discount_rates: (Union[jax.Array, float]): Array of risk-free interest rates
    :param are_calls: (jax.Array) Array of booleans indicating whether options
                                  are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of gamma of the given options.
    """
    return grad(grad(bs_price, argnums=0), argnums=0)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


@jit
def theta_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Optional[jax.Array] = None,
    dtype: Optional[jnp.dtype] = None,
) -> jax.Array:
    """
    Calculate the theta of a european option under the BS model (scalar version)

    :param spots: (Union[jax.Array, float]): Array of current asset prices.
    :param strikes: (Union[jax.Array, float]): Array of option strike prices.
    :param expires: (Union[jax.Array, float]): Array of option expiration times.
    :param vols: (Union[jax.Array, float]): Array of option volatility values.
    :param discount_rates: (Union[jax.Array, float]): Array of risk-free interest rates
    :param are_calls: (jax.Array) Array of booleans indicating whether options
                                  are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of theta of the given options.
    """
    return grad(bs_price, argnums=2)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


@jit
def vega_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Optional[jax.Array] = None,
    dtype: Optional[jnp.dtype] = None,
) -> jax.Array:
    """
    Calculate the vega of a european option under the BS model (scalar version)

    :param spots: (Union[jax.Array, float]): Array of current asset prices.
    :param strikes: (Union[jax.Array, float]): Array of option strike prices.
    :param expires: (Union[jax.Array, float]): Array of option expiration times.
    :param vols: (Union[jax.Array, float]): Array of option volatility values.
    :param discount_rates: (Union[jax.Array, float]): Array of risk-free interest rates
    :param are_calls: (jax.Array) Array of booleans indicating whether options
                                  are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of vega of the given options.
    """
    return grad(bs_price, argnums=3)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


@jit
def rho_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Optional[jax.Array] = None,
    dtype: Optional[jnp.dtype] = None,
) -> jax.Array:
    """
    Calculate the rho of a european option under the BS model (scalar version)

    :param spots: (Union[jax.Array, float]): Array of current asset prices.
    :param strikes: (Union[jax.Array, float]): Array of option strike prices.
    :param expires: (Union[jax.Array, float]): Array of option expiration times.
    :param vols: (Union[jax.Array, float]): Array of option volatility values.
    :param discount_rates: (Union[jax.Array, float]): Array of risk-free interest rates
    :param are_calls: (jax.Array) Array of booleans indicating whether options
                                  are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of rho of the given options.
    """
    return grad(bs_price, argnums=4)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )
