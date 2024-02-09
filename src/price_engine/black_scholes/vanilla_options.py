"""
Black Scholes prices for Vanilla European options
"""
import jax
import jax.numpy as jnp

from ..math import d1, cum_normal
from ..utils import cast_arrays

_SQRT_2 = jnp.sqrt(2.0)


@jax.jit
def _compute_undiscounted_call_prices(spots, strikes, expires, vols, discount_factors):
    vol_sqrt_t = vols * jnp.sqrt(expires)

    _d1 = d1(spots, strikes, vols, expires, discount_factors)
    _d2 = d1 - vol_sqrt_t

    return cum_normal(_d1) * spots - cum_normal(_d2) * strikes


@jax.jit
def _compute_undiscounted_call_prices_f(spots, strikes, expires, vols, discount_factors):
    forwards = discount_factors * spots

    return _compute_undiscounted_call_prices(forwards, strikes, expires, vols, discount_factors)


def vanilla_price(
        spots: jax.Array,
        strikes: jax.Array,
        expires: jax.Array,
        vols: jax.Array,
        discount_rates: jax.Array = None,
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

    if dtype is not None:
        spots = jnp.astype(spots, dtype)
        strikes = jnp.astype(strikes, dtype)
        expires = jnp.astype(expires, dtype)
        vols = jnp.astype(vols, dtype)

    if discount_rates is None:
        discount_rates = jnp.zeros(shape, dtype=dtype)

    if dividend_rates is None:
        dividend_rates = jnp.zeros(shape, dtype=dtype)

    discount_factors = jnp.exp((discount_rates - dividend_rates) * expires)
    forwards = discount_factors * spots

    undiscounted_calls = _compute_undiscounted_call_prices_f(
        spots, strikes, expires, vols, discount_factors
    )

    if are_calls is None:
        return discount_factors * undiscounted_calls

    undiscounted_forwards = forwards - strikes
    undiscouted_puts = undiscounted_calls - undiscounted_forwards
    return discount_factors * jnp.where(are_calls, undiscounted_calls, undiscouted_puts)


def bs_price(
        spots: jax.Array,
        strikes: jax.Array,
        expires: jax.Array,
        vols: jax.Array,
        discount_rates: jax.Array = None,
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
    shape = spots.shape

    [spots, strikes, expires, vols] = cast_arrays([spots, strikes, expires, vols], dtype)

    if discount_rates is None:
        discount_rates = jnp.zeros(shape, dtype=dtype)

    discount_factors = jnp.exp(discount_rates * expires)

    calls = _compute_undiscounted_call_prices(
        spots, strikes, expires, vols, discount_rates
    )

    if are_calls is None:
        return calls

    puts = calls + (strikes * discount_factors) - spots
    return jnp.where(are_calls, calls, puts)


def delta_vanilla(spots: jax.Array,
                  strikes: jax.Array,
                  expires: jax.Array,
                  vols: jax.Array,
                  discount_rates: jax.Array = None,
                  are_calls: jax.Array = None,
                  dtype: jnp.dtype = None,
                  ) -> jax.Array:
    """
    Calculate the delta of a call/put option

    :param spots:
    :param strikes:
    :param expires:
    :param vols:
    :param discount_rates:
    :param are_calls:
    :param dtype:
    :return:
    """
    [spots, strikes, expires, vols] = cast_arrays([spots, strikes, expires, vols], dtype)

    d1s = d1(spots, strikes, vols, expires, discount_rates)

    return cum_normal(d1s) if are_calls is None else jnp.where(are_calls, cum_normal(d1s), cum_normal(d1s) - 1)
