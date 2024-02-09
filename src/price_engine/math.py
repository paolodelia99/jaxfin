import jax
import jax.numpy as jnp
from jax.scipy.special import erf

_SQRT_2 = jnp.sqrt(2.0)


@jax.jit
def d1(spots, strikes, vols, expires, discount_rates):
    vol_sqrt_t = vols * jnp.sqrt(expires)

    return jnp.divide((jnp.log(spots / strikes) + (discount_rates + (vols ** 2 / 2)) * expires), vol_sqrt_t)


@jax.jit
def cum_normal(x):
    return (erf(x / _SQRT_2) + 1) / 2