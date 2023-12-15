import jax.numpy as jnp
from src.price_engine.black_scholes.vanilla_options import _vanilla_price

def test_one_opt_price():
    spot = jnp.float32(100)
    expire = jnp.float32(1.0)
    vol = jnp.float32(0.3)
    strike = jnp.float32(110)
    dtype = jnp.float32

    price = _vanilla_price(spot, strike, expire, vol, dtype=dtype)

    assert price == 8.141014

