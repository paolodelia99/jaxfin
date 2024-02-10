"""
Black '76 prices for options on forwards and futures
"""
from ..math import *
from ..utils import cast_arrays
from ..common import compute_undiscounted_call_prices


def black_price(
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

    [spots, strikes, expires, vols] = cast_arrays([spots, strikes, expires, vols], dtype)

    if discount_rates is None:
        discount_rates = jnp.zeros(shape, dtype=dtype)

    if dividend_rates is None:
        dividend_rates = jnp.zeros(shape, dtype=dtype)

    discount_factors = jnp.exp((discount_rates - dividend_rates) * expires)
    forwards = discount_factors * spots

    undiscounted_calls = compute_undiscounted_call_prices(
        spots * discount_factors, strikes, expires, vols, discount_factors
    )

    if are_calls is None:
        return discount_factors * undiscounted_calls

    undiscounted_forwards = forwards - strikes
    undiscouted_puts = undiscounted_calls - undiscounted_forwards
    return discount_factors * jnp.where(are_calls, undiscounted_calls, undiscouted_puts)
