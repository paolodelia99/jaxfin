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
        dtype: jnp.dtype = None):
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

    undiscounted_calls = _compute_undiscounted_call_prices(spots, strikes, expires, vols, discount_factors)

    if are_calls is None:
        return discount_factors * undiscounted_calls

    undiscounted_forwards = forwards - strikes
    undiscouted_puts = undiscounted_calls - undiscounted_forwards
    return discount_factors * jnp.where(are_calls, undiscounted_calls, undiscouted_puts)


@jax.jit
def _ncdf(x):
    return (erf(x / _SQRT_2) + 1) / 2
