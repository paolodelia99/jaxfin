import jax.numpy as jnp

from jaxfin.price_engine.black_scholes import european_price, delta_european, gamma_european, theta_european, rho_european, vega_european

import logging

import pytest

TOL = 1e-3
DTYPE = jnp.float32

logging.basicConfig(level=logging.INFO)

def test_one_vanilla():
    dtype = jnp.float32
    spot = jnp.array([100], dtype=dtype)
    expire = jnp.array([1.0], dtype=dtype)
    vol = jnp.array([0.3], dtype=dtype)
    strike = jnp.array([110], dtype=dtype)
    risk_free_rate = jnp.array([0.0], dtype=dtype)

    price = european_price(spot, strike, expire, vol, risk_free_rate, dtype=dtype)

    assert price[0] == 8.141014


def test_vanilla_batch_calls():
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
    vols = jnp.array([.3, .25, .4, .2, .1], dtype=DTYPE)
    strikes = jnp.array([120, 120, 120, 120, 120], dtype=DTYPE)
    discount_rates = jnp.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=DTYPE)

    prices = european_price(spots, strikes, expires, vols, discount_rates, dtype=DTYPE)
    expected = jnp.array([5.440567, 1.602787, 3.140933, 5.010391, 4.7853127])

    assert jnp.isclose(prices, expected, atol=TOL).all()


def test_vanilla_batch_mixed():
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
    vols = jnp.array([.3, .25, .4, .2, .1], dtype=DTYPE)
    strikes = jnp.array([120, 75, 75, 120, 120], dtype=DTYPE)
    are_calls = jnp.array([True, False, False, True, True], dtype=jnp.bool_)
    discount_rates = jnp.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=DTYPE)

    prices = european_price(spots, strikes, expires, vols, discount_rates,
                            are_calls=are_calls, dtype=DTYPE)
    expected = jnp.array([5.440567, 2.7794075, 9.942638, 5.010391, 4.7853127])

    assert jnp.isclose(prices, expected, atol=TOL).all()


def test_vanilla_batch_mixed_disc():
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
    vols = jnp.array([.3, .25, .4, .2, .1], dtype=DTYPE)
    strikes = jnp.array([120, 75, 75, 120, 120], dtype=DTYPE)
    are_calls = jnp.array([True, False, False, True, True], dtype=jnp.bool_)
    discount_rates = jnp.array([0.01, 0.0125, 0.02, 0.03, 0.05], dtype=DTYPE)

    prices = european_price(spots, strikes, expires, vols,
                            are_calls=are_calls, discount_rates=discount_rates, dtype=DTYPE)
    expected = jnp.array([5.714386,  2.5328674, 9.19194, 6.155064, 8.165947])

    assert jnp.isclose(prices, expected, atol=TOL).all()

### Test Greeks Calculations ###

class TestDelta:

    def test_delta_bs_float(self):
        spot = jnp.array(100, dtype=DTYPE)
        strike = jnp.array(120, dtype=DTYPE)
        expire = jnp.array(1, dtype=DTYPE)
        vol = jnp.array(0.3, dtype=DTYPE)
        rate = jnp.array(0.0, dtype=DTYPE)
        expected_delta = jnp.array(0.32357, dtype=DTYPE)
        expected_put_delta = jnp.array(-0.67643, dtype=DTYPE)

        logging.info(f"Testing with spot={spot}, strike={strike}, expire={expire}, vol={vol}, rate={rate}")

        call_delta = delta_european(spot, strike, expire, vol, rate)
        put_delta = delta_european(spot, strike, expire, vol, rate, are_calls=False)

        assert jnp.isclose(call_delta, expected_delta, atol=TOL)
        assert jnp.isclose(put_delta, expected_put_delta, atol=TOL)
        assert jnp.isclose(call_delta - 1.0, put_delta, atol=TOL)

    def test_delta_bs(self):
        spots = jnp.array([100, 100, 100, 80, 170], dtype=DTYPE)
        strikes = jnp.array([120, 110, 120, 150, 160], dtype=DTYPE)
        expires = jnp.array([1, 1, 1, 0.5, 0.25], dtype=DTYPE)
        vols = jnp.array([0.3, 0.3, 0.2, 0.5, 0.15], dtype=DTYPE)
        rates = jnp.array([0.0, 0.0, 0.05, 0.02, 0.01], dtype=DTYPE)
        put_flags = jnp.array([False, False, False, False, False], dtype=jnp.bool_)
        e_call_deltas = jnp.array([0.32357, 0.43341, 0.28719, 0.05787, 0.81034], dtype=DTYPE)
        e_put_deltas = jnp.array([-0.67643, -0.56659, -0.71281, -0.94213, -0.18966], dtype=DTYPE)

        logging.info(f"Testing with spots={spots}, strikes={strikes}, expires={expires}, vols={vols}, rates={rates}")

        call_deltas = delta_european(spots, strikes, expires, vols, rates)
        put_deltas = delta_european(spots, strikes, expires, vols, rates, are_calls=put_flags)

        assert jnp.allclose(call_deltas, e_call_deltas, atol=TOL)
        assert jnp.allclose(put_deltas, e_put_deltas, atol=TOL)
        assert jnp.allclose(call_deltas - 1.0, put_deltas, atol=TOL)


class TestDeltaBatch:

    def test_delta_bs_batch(self):
        spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
        expires = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=DTYPE)
        vols = jnp.array([.3, .25, .4, .2, .1], dtype=DTYPE)
        strikes = jnp.array([120, 120, 120, 120, 120], dtype=DTYPE)
        discount_rates = jnp.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=DTYPE)

        call_delta = delta_european(spots, strikes, expires, vols, discount_rates)
        put_flag = jnp.array([False, False, False, False, False], dtype=jnp.bool_)
        put_delta = delta_european(spots, strikes, expires, vols, discount_rates, are_calls=put_flag)

        expected_call = jnp.array([0.323570, 0.152509, 0.207919, 0.36879, 0.51993])
        expected_put = jnp.array([-0.676429, -0.847490, -0.79208, -0.63120, -0.480061])

        assert jnp.isclose(call_delta, expected_call, atol=TOL).all()
        assert jnp.isclose(put_delta, expected_put, atol=TOL).all()
        assert jnp.isclose(call_delta - 1.0, put_delta, atol=TOL).all()

@pytest.mark.parametrize("spot, strike, expire, vol, rate, expected_gamma",
                            [(100, 120, 1, 0.3, 0.0, 0.01198),
                            (100, 110, 1, 0.3, 0.0, 0.01311),
                            (100, 120, 1, 0.2, 0.05, 0.01704),
                            (80, 150, 0.5, 0.5, 0.02, 0.00409),
                            (170, 160, 0.25, 0.15, 0.01, 0.02126)
                            ])
class TestGamma:

    def test_gamma_bs(self, spot, strike, expire, vol, rate, expected_gamma):
        spot = jnp.array([spot], dtype=DTYPE)
        strike = jnp.array([strike], dtype=DTYPE)
        expire = jnp.array([expire], dtype=DTYPE)
        vol = jnp.array([vol], dtype=DTYPE)
        rate = jnp.array([rate], dtype=DTYPE)
        gamma = gamma_european(spot, strike, expire, vol, rate)

        assert jnp.isclose(gamma, expected_gamma, atol=TOL).all()

@pytest.mark.parametrize("spot, strike, expire, vol, rate, e_call_theta, e_put_theta",
                            [(100, 120, 1, 0.3, 0.0, 5.38894, 5.38894),
                            (100, 110, 1, 0.3, 0.0, 5.90058, 5.90058),
                            (100, 120, 1, 0.2, 0.05, 4.68097, -1.02641),
                            (80, 150, 0.5, 0.5, 0.02, 3.35533, 0.38519),
                            (170, 160, 0.25, 0.15, 0.01, 8.17193, 6.57593)
                            ])
class TestTheta:

    def test_theta_bs(self, spot, strike, expire, vol, rate, e_call_theta, e_put_theta):
        spot = jnp.array([spot], dtype=DTYPE)
        strike = jnp.array([strike], dtype=DTYPE)
        expire = jnp.array([expire], dtype=DTYPE)
        vol = jnp.array([vol], dtype=DTYPE)
        rate = jnp.array([rate], dtype=DTYPE)
        put_flag = jnp.array([False], dtype=jnp.bool_)
        e_call_theta = jnp.array([e_call_theta], dtype=DTYPE)
        e_put_theta = jnp.array([e_put_theta], dtype=DTYPE)
        theta = theta_european(spot, strike, expire, vol, rate)
        p_theta = theta_european(spot, strike, expire, vol, rate, are_calls=put_flag)

        assert jnp.isclose(theta, e_call_theta, atol=TOL).all()
        assert jnp.isclose(p_theta, e_put_theta, atol=TOL).all()

@pytest.mark.parametrize("spot, strike, expire, vol, rate, e_call_rho, e_put_rho",
                            [(100, 120, 1, 0.3, 0.0, 26.91644, -93.08356),
                            (100, 110, 1, 0.3, 0.0, 35.19993, -74.80007),
                            (100, 120, 1, 0.2, 0.05, 25.47168, -88.67585),
                            (80, 150, 0.5, 0.5, 0.02, 2.00656, -72.24718),
                            (170, 160, 0.25, 0.15, 0.01, 31.49509, -8.40503)
                            ]) 
class TestRho:

    def test_rho_bs(self, spot, strike, expire, vol, rate, e_call_rho, e_put_rho):
        spot = jnp.array([spot], dtype=DTYPE)
        strike = jnp.array([strike], dtype=DTYPE)
        expire = jnp.array([expire], dtype=DTYPE)
        vol = jnp.array([vol], dtype=DTYPE)
        rate = jnp.array([rate], dtype=DTYPE)
        put_flag = jnp.array([False], dtype=jnp.bool_)
        e_call_rho = jnp.array([e_call_rho], dtype=DTYPE)
        call_rho = rho_european(spot, strike, expire, vol, rate)
        put_rho = rho_european(spot, strike, expire, vol, rate, are_calls=put_flag)

        assert jnp.isclose(call_rho, e_call_rho, atol=TOL).all()
        assert jnp.isclose(put_rho, e_put_rho, atol=TOL).all()

@pytest.mark.parametrize("spot, strike, expire, vol, rate, expected_vega",
                            [(100, 120, 1, 0.3, 0.0, 35.92629),
                            (100, 110, 1, 0.3, 0.0, 39.33717),
                            (100, 120, 1, 0.2, 0.05, 34.07384),
                            (80, 150, 0.5, 0.5, 0.02, 6.55014),
                            (170, 160, 0.25, 0.15, 0.01, 23.04042)
                            ])
class TestVega:
    
        def test_vega_bs(self, spot, strike, expire, vol, rate, expected_vega):
            spot = jnp.array([spot], dtype=DTYPE)
            strike = jnp.array([strike], dtype=DTYPE)
            expire = jnp.array([expire], dtype=DTYPE)
            vol = jnp.array([vol], dtype=DTYPE)
            rate = jnp.array([rate], dtype=DTYPE)
            vega = vega_european(spot, strike, expire, vol, rate)
    
            assert jnp.isclose(vega, expected_vega, atol=TOL).all()
