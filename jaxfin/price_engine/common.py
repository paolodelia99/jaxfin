from .math import *


def _compute_d1_d2(spots, strikes, expires, vols, discount_rates):
    vol_sqrt_t = vols * jnp.sqrt(expires)

    _d1 = d1(spots, strikes, vols, expires, discount_rates)

    return [_d1, _d1 - vol_sqrt_t]


@jax.jit
def compute_undiscounted_call_prices(spots, strikes, expires, vols, discount_rates):
    [_d1, _d2] = _compute_d1_d2(spots, strikes, expires, vols, discount_rates)

    return cum_normal(_d1) * spots - cum_normal(_d2) * strikes


@jax.jit
def compute_discounted_call_prices(spots, strikes, expires, vols, discount_rates):
    [_d1, _d2] = _compute_d1_d2(spots, strikes, expires, vols, discount_rates)

    return cum_normal(_d1) * spots - cum_normal(_d2) * strikes * jnp.exp(
        (-discount_rates) * expires
    )
