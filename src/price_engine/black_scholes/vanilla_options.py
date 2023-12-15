"""
Black Scholes prices for Vanilla European options
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erf

_SQRT_2 = jnp.sqrt(2.0)


def _vanilla_price(
        spot,
        strike,
        expire,
        vol,
        discount_rate=None,
        dividend_rate=None,
        is_call: jnp.bool_ = True,
        dtype: jnp.dtype = None):
    if dtype is not None:
        spot = jnp.astype(spot, dtype)
        strike = jnp.astype(strike, dtype)
        expire = jnp.astype(expire, dtype)
        vol = jnp.astype(vol, dtype)

    if discount_rate is None:
        discount_rate = jnp.float32(0.0)

    if dividend_rate is None:
        dividend_rate = jnp.float32(0.0)

    _D = jnp.exp((discount_rate - dividend_rate) * expire)
    forward = _D * spot

    vol_sqrt_t = vol * jnp.sqrt(expire)

    d1 = jnp.divide(jnp.log(forward / strike), vol_sqrt_t) + vol_sqrt_t / 2
    d2 = d1 - vol_sqrt_t

    call_price = _D * (_ncdf(d1) * forward - _ncdf(d2) * strike)

    if is_call:
        return call_price
    else:
        return call_price + _D * strike - spot


vanilla_price = jax.vmap(_vanilla_price)


def _ncdf(x):
    return (erf(x / _SQRT_2) + 1) / 2
