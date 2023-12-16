import jax.numpy as jnp

from src.price_engine.black_scholes.vanilla_options import vanilla_price


def test_one_vanilla():
    dtype = jnp.float32
    spot = jnp.array([100], dtype=dtype)
    expire = jnp.array([1.0], dtype=dtype)
    vol = jnp.array([0.3], dtype=dtype)
    strike = jnp.array([110], dtype=dtype)

    price = vanilla_price(spot, strike, expire, vol, dtype=dtype)

    assert price[0] == 8.141014

def test_vanilla_batch_calls():
    dtype = jnp.float32
    spots = jnp.array([100, 90, 80, 110, 120], dtype=dtype)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
    vols = jnp.array([.3, .25, .4, .2, .1], dtype=dtype)
    strikes = jnp.array([120, 120, 120, 120, 120], dtype=dtype)

    prices = vanilla_price(spots, strikes, expires, vols, dtype=dtype)
    expected = jnp.array([5.440567,  1.602787,  3.140933,  5.010391, 4.7853127])

    assert jnp.array_equal(prices, expected)

def test_vanilla_batch_mixed():
    dtype = jnp.float32
    spots = jnp.array([100, 90, 80, 110, 120], dtype=dtype)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
    vols = jnp.array([.3, .25, .4, .2, .1], dtype=dtype)
    strikes = jnp.array([120, 75, 75, 120, 120], dtype=dtype)
    are_calls = jnp.array([True, False, False, True, True], dtype=jnp.bool_)

    prices = vanilla_price(spots, strikes, expires, vols,
                           are_calls=are_calls, dtype=dtype)
    expected = jnp.array([5.440567,  2.7794075,  9.942638,  5.010391, 4.7853127])

    assert jnp.array_equal(prices, expected)