import jax.numpy as jnp
import pytest

from jaxfin.price_engine.black import (
    future_option_delta,
    future_option_gamma,
    future_option_price,
)
from jaxfin.price_engine.utils.vect import get_vfunction

TOL = 1e-3
DTYPE = jnp.float32


def test_one_future_option_float():
    spot = jnp.array(100, dtype=DTYPE)
    expire = jnp.array(1.0, dtype=DTYPE)
    vol = jnp.array(0.3, dtype=DTYPE)
    strike = jnp.array(110, dtype=DTYPE)
    risk_free_rate = jnp.array(0.0, dtype=DTYPE)

    price = future_option_price(spot, strike, expire, vol, risk_free_rate, dtype=DTYPE)

    assert price == 8.141014


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
    vols = jnp.array([0.3, 0.25, 0.4, 0.2, 0.1], dtype=DTYPE)
    strikes = jnp.array([120, 120, 120, 120, 120], dtype=DTYPE)
    discount_rates = jnp.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=DTYPE)

    prices = future_option_price(
        spots, strikes, expires, vols, discount_rates, dtype=DTYPE
    )
    expected = jnp.array([5.440567, 1.602787, 3.140933, 5.010391, 4.7853127])

    assert jnp.isclose(prices, expected, atol=TOL).all()


def test_foption_batch_mixed():
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
    vols = jnp.array([0.3, 0.25, 0.4, 0.2, 0.1], dtype=DTYPE)
    strikes = jnp.array([120, 75, 75, 120, 120], dtype=DTYPE)
    are_calls = jnp.array([True, False, False, True, True], dtype=jnp.bool_)
    discount_rates = jnp.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=DTYPE)

    prices = future_option_price(
        spots, strikes, expires, vols, discount_rates, are_calls=are_calls, dtype=DTYPE
    )
    expected = jnp.array([5.440567, 2.7794075, 9.942638, 5.010391, 4.7853127])

    assert jnp.isclose(prices, expected, atol=TOL).all()


def test_foption_batch_mixed_disc():
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
    vols = jnp.array([0.3, 0.25, 0.4, 0.2, 0.1], dtype=DTYPE)
    strikes = jnp.array([120, 75, 75, 120, 120], dtype=DTYPE)
    are_calls = jnp.array([True, False, False, True, True], dtype=jnp.bool_)
    discount_rates = jnp.array([0.01, 0.0125, 0.02, 0.03, 0.05], dtype=DTYPE)

    prices = future_option_price(
        spots,
        strikes,
        expires,
        vols,
        are_calls=are_calls,
        discount_rates=discount_rates,
        dtype=DTYPE,
    )
    expected = jnp.array([5.7082324, 2.5256162, 9.177393, 6.055744, 7.7550125])

    assert jnp.isclose(prices, expected, atol=TOL).all()


class TestDelta:
    def test_delta_float(self):
        spot = jnp.array(100, dtype=DTYPE)
        strike = jnp.array(120, dtype=DTYPE)
        expire = jnp.array(1, dtype=DTYPE)
        vol = jnp.array(0.3, dtype=DTYPE)
        rate = jnp.array(0.0, dtype=DTYPE)
        expected_delta = jnp.array(0.32357, dtype=DTYPE)
        expected_put_delta = jnp.array(-0.67643, dtype=DTYPE)

        call_delta = future_option_delta(spot, strike, expire, vol, rate)
        put_delta = future_option_delta(
            spot, strike, expire, vol, rate, are_calls=False
        )

        assert jnp.isclose(call_delta, expected_delta, atol=TOL)
        assert jnp.isclose(put_delta, expected_put_delta, atol=TOL)

    def test_delta(self):
        spots = jnp.array([100, 100, 100, 80, 170], dtype=DTYPE)
        strikes = jnp.array([120, 110, 120, 150, 160], dtype=DTYPE)
        expires = jnp.array([1, 1, 1, 0.5, 0.25], dtype=DTYPE)
        vols = jnp.array([0.3, 0.3, 0.2, 0.5, 0.15], dtype=DTYPE)
        rates = jnp.array([0.0, 0.0, 0.05, 0.02, 0.01], dtype=DTYPE)
        dividends = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=DTYPE)
        call_flags = jnp.array([True, True, True, True, True], dtype=jnp.bool_)
        put_flags = jnp.array([False, False, False, False, False], dtype=jnp.bool_)
        e_call_deltas = jnp.array(
            [0.32357, 0.43340, 0.28024, 0.05777, 0.81046], dtype=DTYPE
        )
        e_put_deltas = jnp.array(
            [-0.67643, -0.56659, -0.71975, -0.94222, -0.18953], dtype=DTYPE
        )
        vmap_delta = get_vfunction(
            future_option_delta,
            spots,
            strikes,
            expires,
            vols,
            rates,
            dividends,
            are_calls=put_flags,
        )

        call_deltas = vmap_delta(
            spots, strikes, expires, vols, rates, dividends, call_flags
        )
        put_deltas = vmap_delta(
            spots, strikes, expires, vols, rates, dividends, put_flags
        )

        assert jnp.allclose(call_deltas, e_call_deltas, atol=TOL)
        assert jnp.allclose(put_deltas, e_put_deltas, atol=TOL)


@pytest.mark.parametrize(
    "spot, strike, expire, vol, rate, expected_gamma",
    [
        (100, 120, 1, 0.3, 0.0, 0.01197),
        (100, 110, 1, 0.3, 0.0, 0.013112),
        (100, 120, 1, 0.2, 0.01, 0.01523321),
        (80, 150, 0.5, 0.5, 0.02, 0.00417),
        (170, 160, 0.25, 0.15, 0.01, 0.02136),
    ],
)
class TestGamma:
    def test_gamma_bs(self, spot, strike, expire, vol, rate, expected_gamma):
        spot = jnp.array(spot, dtype=DTYPE)
        strike = jnp.array(strike, dtype=DTYPE)
        expire = jnp.array(expire, dtype=DTYPE)
        vol = jnp.array(vol, dtype=DTYPE)
        rate = jnp.array(rate, dtype=DTYPE)
        dividends = jnp.array(0.0, dtype=DTYPE)
        put_flag = jnp.array(False, dtype=jnp.bool_)
        expected_gamma = jnp.array(expected_gamma, dtype=DTYPE)
        call_gamma = future_option_gamma(spot, strike, expire, vol, rate, dividends)
        put_gamma = future_option_gamma(
            spot, strike, expire, vol, rate, dividends, are_calls=put_flag
        )

        assert jnp.isclose(call_gamma, expected_gamma, atol=TOL).all()
        assert jnp.isclose(put_gamma, call_gamma, atol=TOL).all()
