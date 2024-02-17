"""
Black Scholes prices for Vanilla European options
"""
import jax
import jax.numpy as jnp

from ..common import compute_discounted_call_prices
from ..math import cum_normal, d1, density_normal
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


def delta_vanilla(
    spots: jax.Array,
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
    [spots, strikes, expires, vols] = cast_arrays(
        [spots, strikes, expires, vols], dtype
    )

    d1s = d1(spots, strikes, vols, expires, discount_rates)

    return (
        cum_normal(d1s)
        if are_calls is None
        else jnp.where(are_calls, cum_normal(d1s), cum_normal(d1s) - 1)
    )


def gamma_vanilla(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Calculate the gamma of a european option

    :param spots:
    :param strikes:
    :param expires:
    :param vols:
    :param discount_rates:
    :param dtype:
    :return:
    """
    [spots, strikes, expires, vols] = cast_arrays(
        [spots, strikes, expires, vols], dtype
    )

    d1s = d1(spots, strikes, vols, expires, discount_rates)

    return density_normal(d1s) / (spots * vols * jnp.sqrt(expires))
