"""
Black '76 prices for options on forwards and futures
"""
from typing import Union

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

from ..common import compute_undiscounted_call_prices
from ..utils import cast_arrays


def black_price(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    dividend_rates: jax.Array = None,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Compute the option prices for european options using the Black '76 model.

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
    shape = spots.shape

    [spots, strikes, expires, vols] = cast_arrays(
        [spots, strikes, expires, vols], dtype
    )

    if dividend_rates is None:
        dividend_rates = jnp.zeros(shape, dtype=dtype)

    discount_factors = jnp.exp((discount_rates - dividend_rates) * expires)
    forwards = discount_factors * spots

    undiscounted_calls = compute_undiscounted_call_prices(
        forwards, strikes, expires, vols, discount_rates
    )

    if are_calls is None:
        return discount_factors * undiscounted_calls

    undiscounted_forwards = forwards - strikes
    undiscouted_puts = undiscounted_calls - undiscounted_forwards
    return jnp.exp((-1 * discount_rates) * expires) * jnp.where(
        are_calls, undiscounted_calls, undiscouted_puts
    )


@jit
def _delta_black(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    dividend_rates: jax.Array,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Compute the option deltas for european options using the Black '76 model.

    :param spots: (jax.Array): Array of current asset prices.
    :param strikes: (jax.Array): Array of option strike prices.
    :param expires: (jax.Array): Array of option expiration times.
    :param vols: (jax.Array): Array of option volatility values.
    :param discount_rates: (jax.Array): Array of risk-free interest rates. Defaults to None.
    :param dividend_rates: (jax.Array): Array of dividend rates. Defaults to None.
    :param are_calls: (jax.Array): Array of booleans indicating whether options are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of computed option deltas.
    """
    return grad(black_price, argnums=0)(
        spots, strikes, expires, vols, discount_rates, dividend_rates, are_calls, dtype
    )


@jit
def _gamma_black(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    dividend_rates: jax.Array,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Compute the option gammas for european options using the Black '76 model.

    :param spots: (jax.Array): Array of current asset prices.
    :param strikes: (jax.Array): Array of option strike prices.
    :param expires: (jax.Array): Array of option expiration times.
    :param vols: (jax.Array): Array of option volatility values.
    :param discount_rates: (jax.Array): Array of risk-free interest rates. Defaults to None.
    :param dividend_rates: (jax.Array): Array of dividend rates. Defaults to None.
    :param are_calls: (jax.Array): Array of booleans indicating whether options are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of computed option gammas.
    """
    return grad(grad(black_price, argnums=0), argnums=0)(
        spots, strikes, expires, vols, discount_rates, dividend_rates, are_calls, dtype
    )


def delta_black(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    dividend_rates: Union[jax.Array, float] = None,
    are_calls: Union[jax.Array, bool] = None,
    dtype: jnp.dtype = None,
) -> Union[jax.Array, float]:
    """
    Compute the option deltas for european options using the Black '76 model. (vectorized)

    :param spots: (Union[jax.Array, float]): Current asset price or array of prices.
    :param strikes: (Union[jax.Array, float]): Option strike price or array of prices.
    :param expires: (Union[jax.Array, float]): Option expiration time or array of times.
    :param vols: (Union[jax.Array, float]): Option volatility value or array of values.
    :param discount_rates: (Union[jax.Array, float]): Risk-free interest rate or array of rates.
    :param dividend_rates: (Union[jax.Array, float]): Dividend rate or array of rates. Defaults to None.
    :param are_calls: (Union[jax.Array, bool]): Boolean indicating whether option is a
                                                call or put, or array of booleans.
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (Union[jax.Array, float]): Delta of the given option or array of deltas.
    """
    if jnp.isscalar(spots) or spots.shape == ():
        return _delta_black(
            spots,
            strikes,
            expires,
            vols,
            discount_rates,
            dividend_rates,
            are_calls,
            dtype,
        )

    return jit(vmap(_delta_black, in_axes=(0, 0, 0, 0, 0, 0, 0, None)))(
        spots,
        strikes,
        expires,
        vols,
        discount_rates,
        dividend_rates,
        are_calls,
        dtype,
    )


def gamma_black(
    spots: Union[jax.Array, float],
    strikes: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    vols: Union[jax.Array, float],
    discount_rates: Union[jax.Array, float],
    dividend_rates: Union[jax.Array, float] = None,
    are_calls: Union[jax.Array, bool] = None,
    dtype: jnp.dtype = None,
) -> Union[jax.Array, float]:
    """
    Compute the option gammas for european options using the Black '76 model. (vectorized)

    :param spots: (Union[jax.Array, float]): Current asset price or array of prices.
    :param strikes: (Union[jax.Array, float]): Option strike price or array of prices.
    :param expires: (Union[jax.Array, float]): Option expiration time or array of times.
    :param vols: (Union[jax.Array, float]): Option volatility value or array of values.
    :param discount_rates: (Union[jax.Array, float]): Risk-free interest rate or array of rates.
    :param dividend_rates: (Union[jax.Array, float]): Dividend rate or array of rates. Defaults to None.
    :param are_calls: (Union[jax.Array, bool]): Boolean indicating whether option is a
                                                call or put, or array of booleans.
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (Union[jax.Array, float]): Gamma of the given option or array of gammas.
    """
    if jnp.isscalar(spots) or spots.shape == ():
        return _gamma_black(
            spots,
            strikes,
            expires,
            vols,
            discount_rates,
            dividend_rates,
            are_calls,
            dtype,
        )

    return jit(vmap(_gamma_black, in_axes=(0, 0, 0, 0, 0, 0, 0, None)))(
        spots,
        strikes,
        expires,
        vols,
        discount_rates,
        dividend_rates,
        are_calls,
        dtype,
    )
