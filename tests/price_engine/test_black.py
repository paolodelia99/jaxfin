import jax.numpy as jnp

from jaxfin.price_engine.black import future_option_price

TOL = 1e-3
DTYPE = jnp.float32


def test_one_future_option():
    spot = jnp.array([100], dtype=DTYPE)
    expire = jnp.array([1.0], dtype=DTYPE)
    vol = jnp.array([0.3], dtype=DTYPE)
    strike = jnp.array([110], dtype=DTYPE)
    risk_free_rate = jnp.array([0.0], dtype=DTYPE)

    price = future_option_price(spot, strike, expire, vol, risk_free_rate, dtype=DTYPE)

    assert price[0] == 8.141014


def test_foption_batch_calls():
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
    vols = jnp.array([.3, .25, .4, .2, .1], dtype=DTYPE)
    strikes = jnp.array([120, 120, 120, 120, 120], dtype=DTYPE)
    discount_rates = jnp.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=DTYPE)

    prices = future_option_price(spots, strikes, expires, vols, discount_rates,
                                  dtype=DTYPE)
    expected = jnp.array([5.440567, 1.602787, 3.140933, 5.010391, 4.7853127])

    assert jnp.isclose(prices, expected, atol=TOL).all()


def test_foption_batch_mixed():
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
    vols = jnp.array([.3, .25, .4, .2, .1], dtype=DTYPE)
    strikes = jnp.array([120, 75, 75, 120, 120], dtype=DTYPE)
    are_calls = jnp.array([True, False, False, True, True], dtype=jnp.bool_)
    discount_rates = jnp.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=DTYPE)

    prices = future_option_price(spots, strikes, expires, vols, discount_rates,
                                 are_calls=are_calls, dtype=DTYPE)
    expected = jnp.array([5.440567, 2.7794075, 9.942638, 5.010391, 4.7853127])

    assert jnp.isclose(prices, expected, atol=TOL).all()


def test_foption_batch_mixed_disc():
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
    vols = jnp.array([.3, .25, .4, .2, .1], dtype=DTYPE)
    strikes = jnp.array([120, 75, 75, 120, 120], dtype=DTYPE)
    are_calls = jnp.array([True, False, False, True, True], dtype=jnp.bool_)
    discount_rates = jnp.array([0.01, 0.0125, 0.02, 0.03, 0.05], dtype=DTYPE)

    prices = future_option_price(spots, strikes, expires, vols,
                                 are_calls=are_calls, discount_rates=discount_rates, dtype=DTYPE)
    expected = jnp.array([5.7082324, 2.5256162, 9.177393, 6.055744, 7.7550125])

    assert jnp.isclose(prices, expected, atol=TOL).all()
