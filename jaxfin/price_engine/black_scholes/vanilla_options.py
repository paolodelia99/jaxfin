"""
Black Scholes prices for Vanilla European options
"""
from typing import Union

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

from ..common import compute_discounted_call_prices
from ..utils import cast_arrays


def bs_price(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Compute the option prices for european options using the Black-Scholes model.

    :param spots: (jax.Array): Array of current asset prices.
    :param strikes: (jax.Array): Array of option strike prices.
    :param expires: (jax.Array): Array of option expiration times.
    :param vols: (jax.Array): Array of option volatility values.
    :param discount_rates: (jax.Array): Array of risk-free interest rates. Defaults to None.
    :param dividend_rates: (jax.Array): Array of dividend rates. Defaults to None.
    :param are_calls: (jax.Array): Array of booleans indicating whether options are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of computed option prices.
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
def _delta_vanilla(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Calculate the delta of a call/put option under the BS model (scalar version)

    :param spots: (jax.Array): Array of current asset prices.
    :param strikes: (jax.Array): Array of option strike prices.
    :param expires: (jax.Array): Array of option expiration times.
    :param vols: (jax.Array): Array of option volatility values.
    :param discount_rates: (jax.Array): Array of risk-free interest rates.
    :param are_calls: (jax.Array): Array of booleans indicating whether options
                                   are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of delta of the given options.
    """
    return grad(bs_price, argnums=0)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


@jit
def _gamma_vanilla(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Calculate the gamma of a european option under the BS model (scalar version)

    :param spots: (jax.Array): Array of current asset prices.
    :param strikes: (jax.Array): Array of option strike prices.
    :param expires: (jax.Array): Array of option expiration times.
    :param vols: (jax.Array): Array of option volatility values.
    :param discount_rates: (jax.Array): Array of risk-free interest rates
    :param are_calls: (jax.Array) Array of booleans indicating whether options
                                  are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of gamma of the given options.
    """
    return grad(grad(bs_price, argnums=0), argnums=0)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


@jit
def _theta_vanilla(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Calculate the theta of a european option under the BS model (scalar version)

    :param spots: (jax.Array): Array of current asset prices.
    :param strikes: (jax.Array): Array of option strike prices.
    :param expires: (jax.Array): Array of option expiration times.
    :param vols: (jax.Array): Array of option volatility values.
    :param discount_rates: (jax.Array): Array of risk-free interest rates
    :param are_calls: (jax.Array) Array of booleans indicating whether options
                                  are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of theta of the given options.
    """
    return grad(bs_price, argnums=2)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


@jit
def _vega_vanilla(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Calculate the vega of a european option under the BS model (scalar version)

    :param spots: (jax.Array): Array of current asset prices.
    :param strikes: (jax.Array): Array of option strike prices.
    :param expires: (jax.Array): Array of option expiration times.
    :param vols: (jax.Array): Array of option volatility values.
    :param discount_rates: (jax.Array): Array of risk-free interest rates
    :param are_calls: (jax.Array) Array of booleans indicating whether options
                                  are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of vega of the given options.
    """
    return grad(bs_price, argnums=3)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


@jit
def _rho_vanilla(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Calculate the rho of a european option under the BS model (scalar version)

    :param spots: (jax.Array): Array of current asset prices.
    :param strikes: (jax.Array): Array of option strike prices.
    :param expires: (jax.Array): Array of option expiration times.
    :param vols: (jax.Array): Array of option volatility values.
    :param discount_rates: (jax.Array): Array of risk-free interest rates
    :param are_calls: (jax.Array) Array of booleans indicating whether options
                                  are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of rho of the given options.
    """
    return grad(bs_price, argnums=4)(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


# Vectorize version of the functions since we are dealing with arrays


def delta_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Union[jax.Array, bool] = None,
    dtype: jnp.dtype = None,
) -> Union[jax.Array, float]:
    """
    Calculate the delta of a call/put option under the BS model (vectorized)

    :param spots: (Union[jax.Array, float]): Current asset price or array of prices.
    :param strikes: (Union[jax.Array, float]): Option strike price or array of prices.
    :param expires: (Union[jax.Array, float]): Option expiration time or array of times.
    :param vols: (Union[jax.Array, float]): Option volatility value or array of values.
    :param discount_rates: (Union[jax.Array, float]): Risk-free interest rate or array of rates.
    :param are_calls: (Union[jax.Array, bool]): Boolean indicating whether option is a call or
                                                put, or array of booleans.
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (Union[jax.Array, float]): Delta of the given option or array of deltas.
    """
    if jnp.isscalar(spots) or spots.shape == ():
        return _delta_vanilla(
            spots, strikes, expires, vols, discount_rates, are_calls, dtype
        )

    return jit(vmap(_delta_vanilla, in_axes=(0, 0, 0, 0, 0, 0, None)))(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


def gamma_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Union[jax.Array, bool] = None,
    dtype: jnp.dtype = None,
) -> Union[jax.Array, float]:
    """
    Calculate the gamma of a european option under the BS model (vectorized)

    :param spots: (Union[jax.Array, float]): Current asset price or array of prices.
    :param strikes: (Union[jax.Array, float]): Option strike price or array of prices.
    :param expires: (Union[jax.Array, float]): Option expiration time or array of times.
    :param vols: (Union[jax.Array, float]): Option volatility value or array of values.
    :param discount_rates: (Union[jax.Array, float]): Risk-free interest rate or array of rates.
    :param are_calls: (Union[jax.Array, bool]): Boolean indicating whether option is a
                                                call or put, or array of booleans.
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (Union[jax.Array, float]): Gamma of the given option or array of gammas.
    """
    if jnp.isscalar(spots) or spots.shape == ():
        return _gamma_vanilla(
            spots, strikes, expires, vols, discount_rates, are_calls, dtype
        )

    return jit(vmap(_gamma_vanilla, in_axes=(0, 0, 0, 0, 0, 0, None)))(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


def theta_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Union[jax.Array, bool] = None,
    dtype: jnp.dtype = None,
) -> Union[jax.Array, float]:
    """
    Calculate the theta of a european option under the BS model (vectorized)

    :param spots: (Union[jax.Array, float]): Current asset price or array of prices.
    :param strikes: (Union[jax.Array, float]): Option strike price or array of prices.
    :param expires: (Union[jax.Array, float]): Option expiration time or array of times.
    :param vols: (Union[jax.Array, float]): Option volatility value or array of values.
    :param discount_rates: (Union[jax.Array, float]): Risk-free interest rate or array of rates.
    :param are_calls: (Union[jax.Array, bool]): Boolean indicating whether option is a
                                                call or put, or array of booleans.
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (Union[jax.Array, float]): Theta of the given option or array of thetas.
    """
    if jnp.isscalar(spots) or spots.shape == ():
        return _theta_vanilla(
            spots, strikes, expires, vols, discount_rates, are_calls, dtype
        )

    return jit(vmap(_theta_vanilla, in_axes=(0, 0, 0, 0, 0, 0, None)))(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


def rho_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Union[jax.Array, bool] = None,
    dtype: jnp.dtype = None,
) -> Union[jax.Array, float]:
    """
    Calculate the rho of a european option under the BS model (vectorized)

    :param spots: (Union[jax.Array, float]): Current asset price or array of prices.
    :param strikes: (Union[jax.Array, float]): Option strike price or array of prices.
    :param expires: (Union[jax.Array, float]): Option expiration time or array of times.
    :param vols: (Union[jax.Array, float]): Option volatility value or array of values.
    :param discount_rates: (Union[jax.Array, float]): Risk-free interest rate or array of rates.
    :param are_calls: (Union[jax.Array, bool]): Boolean indicating whether option is a
                                                call or put, or array of booleans.
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (Union[jax.Array, float]): Rho of the given option or array of rhos.
    """
    if jnp.isscalar(spots) or spots.shape == ():
        return _rho_vanilla(
            spots, strikes, expires, vols, discount_rates, are_calls, dtype
        )

    return jit(vmap(_rho_vanilla, in_axes=(0, 0, 0, 0, 0, 0, None)))(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )


def vega_vanilla(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    are_calls: Union[jax.Array, bool] = None,
    dtype: jnp.dtype = None,
) -> Union[jax.Array, float]:
    """
    Calculate the vega of a european option under the BS model (vectorized)

    :param spots: (Union[jax.Array, float]): Current asset price or array of prices.
    :param strikes: (Union[jax.Array, float]): Option strike price or array of prices.
    :param expires: (Union[jax.Array, float]): Option expiration time or array of times.
    :param vols: (Union[jax.Array, float]): Option volatility value or array of values.
    :param discount_rates: (Union[jax.Array, float]): Risk-free interest rate or array of rates.
    :param are_calls: (Union[jax.Array, bool]): Boolean indicating whether option is a
                                                call or put, or array of booleans.
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (Union[jax.Array, float]): Vega of the given option or array of vegas.
    """
    if jnp.isscalar(spots):
        return _vega_vanilla(
            spots, strikes, expires, vols, discount_rates, are_calls, dtype
        )

    return jit(vmap(_vega_vanilla, in_axes=(0, 0, 0, 0, 0, 0, None)))(
        spots, strikes, expires, vols, discount_rates, are_calls, dtype
    )
