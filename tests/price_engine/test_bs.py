import jax.numpy as jnp

from src.price_engine.black_scholes import european_price


def test_one_vanilla():
    dtype = jnp.float32
    spot = jnp.array([100], dtype=dtype)
    expire = jnp.array([1.0], dtype=dtype)
    vol = jnp.array([0.3], dtype=dtype)
    strike = jnp.array([110], dtype=dtype)

    price = european_price(spot, strike, expire, vol, dtype=dtype)

    assert price[0] == 8.141014
