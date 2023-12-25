"""
Black Scholes prices for Vanilla European options
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erf

_SQRT_2 = jnp.sqrt(2.0)


@jax.jit
def _compute_undiscounted_call_prices(spots, strikes, expires, vols, discount_factors):
    forwards = discount_factors * spots

    vol_sqrt_t = vols * jnp.sqrt(expires)

    d1 = jnp.divide(jnp.log(forwards / strikes), vol_sqrt_t) + vol_sqrt_t / 2
    d2 = d1 - vol_sqrt_t

    return _ncdf(d1) * forwards - _ncdf(d2) * strikes


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

    undiscounted_calls = _compute_undiscounted_call_prices(
        spots, strikes, expires, vols, discount_factors
    )

    if are_calls is None:
        return discount_factors * undiscounted_calls

    undiscounted_forwards = forwards - strikes
    undiscouted_puts = undiscounted_calls - undiscounted_forwards
    return discount_factors * jnp.where(are_calls, undiscounted_calls, undiscouted_puts)


@jax.jit
def _ncdf(x):
    return (erf(x / _SQRT_2) + 1) / 2
