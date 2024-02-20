import jax.numpy as jnp

import pytest

from jaxfin.price_engine.black import future_option_price, future_option_delta, future_option_gamma

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

@pytest.mark.parametrize("spot, strike, expire, vol, rate, e_call_delta, e_put_delta",
                         [(100, 120, 1, 0.3, 0.0, 0.32357, -0.67643),
                          (100, 110, 1, 0.3, 0.0, 0.43341, -0.56659),
                          (100, 120, 1, 0.2, 0.05, 0.30971, -0.71975),
                          (80, 150, 0.5, 0.5, 0.02, 0.05894, -0.94222),
                          (170, 160, 0.25, 0.15, 0.01, 0.81452, -0.18953)
                        ])
class TestDelta:

    def test_delta_bs(self, spot, strike, expire, vol, rate, e_call_delta, e_put_delta):
        spot = jnp.array([spot], dtype=DTYPE)
        strike = jnp.array([strike], dtype=DTYPE)
        expire = jnp.array([expire], dtype=DTYPE)
        vol = jnp.array([vol], dtype=DTYPE)
        rate = jnp.array([rate], dtype=DTYPE)
        dividends = jnp.array([0.0], dtype=DTYPE)
        put_flag = jnp.array([False], dtype=jnp.bool_)
        e_call_delta = jnp.array([e_call_delta], dtype=DTYPE)
        call_delta = future_option_delta(spot, strike, expire, vol, rate, dividends)
        put_delta = future_option_delta(spot, strike, expire, vol, rate, dividends, are_calls=put_flag)

        assert jnp.isclose(call_delta, e_call_delta, atol=TOL).all()
        assert jnp.isclose(put_delta, e_put_delta, atol=TOL).all()


@pytest.mark.parametrize("spot, strike, expire, vol, rate, expected_gamma",
                            [(100, 120, 1, 0.3, 0.0, 0.01197),
                            (100, 110, 1, 0.3, 0.0, 0.013112),
                            (100, 120, 1, 0.2, 0.01, 0.01523321),
                            (80, 150, 0.5, 0.5, 0.02, 0.00417),
                            (170, 160, 0.25, 0.15, 0.01, 0.02136)
                            ])
class TestGamma:

    def test_gamma_bs(self, spot, strike, expire, vol, rate, expected_gamma):
        spot = jnp.array([spot], dtype=DTYPE)
        strike = jnp.array([strike], dtype=DTYPE)
        expire = jnp.array([expire], dtype=DTYPE)
        vol = jnp.array([vol], dtype=DTYPE)
        rate = jnp.array([rate], dtype=DTYPE)
        dividends = jnp.array([0.0], dtype=DTYPE)
        put_flag = jnp.array([False], dtype=jnp.bool_)
        expected_gamma = jnp.array([expected_gamma], dtype=DTYPE)
        call_gamma = future_option_gamma(spot, strike, expire, vol, rate, dividends)
        put_gamma = future_option_gamma(spot, strike, expire, vol, rate, dividends, are_calls=put_flag)

        assert jnp.isclose(call_gamma, expected_gamma, atol=TOL).all()
        assert jnp.isclose(put_gamma, call_gamma, atol=TOL).all()
