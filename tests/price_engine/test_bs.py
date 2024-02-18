import jax.numpy as jnp

from jaxfin.price_engine.black_scholes import european_price, delta_european, gamma_european, theta_european, rho_european, vega_european

import pytest

TOL = 1e-3
DTYPE = jnp.float32


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

@pytest.mark.parametrize("spot, strike, expire, vol, rate, e_call_delta, e_put_delta",
                         [(100, 120, 1, 0.3, 0.0, 0.32357, -0.67643),
                          (100, 110, 1, 0.3, 0.0, 0.43341, -0.56659),
                          (100, 120, 1, 0.2, 0.05, 0.28719, -0.71281),
                          (80, 150, 0.5, 0.5, 0.02, 0.05787, -0.94213),
                          (170, 160, 0.25, 0.15, 0.01, 0.81034, -0.18966)
                        ])
class TestDelta:

    def test_delta_bs(self, spot, strike, expire, vol, rate, e_call_delta, e_put_delta):
        spot = jnp.array([spot], dtype=DTYPE)
        strike = jnp.array([strike], dtype=DTYPE)
        expire = jnp.array([expire], dtype=DTYPE)
        vol = jnp.array([vol], dtype=DTYPE)
        rate = jnp.array([rate], dtype=DTYPE)
        put_flag = jnp.array([False], dtype=jnp.bool_)
        e_call_delta = jnp.array([e_call_delta], dtype=DTYPE)
        call_delta = delta_european(spot, strike, expire, vol, rate)
        put_delta = delta_european(spot, strike, expire, vol, rate, are_calls=put_flag)

        assert jnp.isclose(call_delta, e_call_delta, atol=TOL).all()
        assert jnp.isclose(put_delta, e_put_delta, atol=TOL).all()
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
        theta = theta_european(spot, strike, expire, vol, rate)

        assert jnp.isclose(theta, e_call_theta, atol=TOL).all()

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
