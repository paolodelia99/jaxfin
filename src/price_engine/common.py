from .math import *


@jax.jit
def compute_undiscounted_call_prices(spots, strikes, expires, vols, discount_factors):
    vol_sqrt_t = vols * jnp.sqrt(expires)

    _d1 = d1(spots, strikes, vols, expires, discount_factors)
    _d2 = _d1 - vol_sqrt_t

    return cum_normal(_d1) * spots - cum_normal(_d2) * strikes
