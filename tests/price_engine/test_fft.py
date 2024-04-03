import jax.numpy as jnp
from jax import vmap

from jaxfin.price_engine.fft import (
    delta_call_fourier,
    fourier_inv_call,
    fourier_inv_put,
)

TOL = 1e-3
DTYPE = jnp.float32


def test_one_vanilla_call():
    spot = 100
    strike = 110
    expire = 1.0
    v0 = 0.2
    theta = 0.3
    rate = 0.0
    sigma = 0.3
    kappa = 2.0
    rho = -0.6

    price = fourier_inv_call(spot, strike, expire, v0, rate, theta, sigma, kappa, rho)
    expected_price = jnp.asarray(15.846119, dtype=DTYPE)

    assert jnp.array_equal(price, expected_price)


def test_one_vanilla_put():
    spot = 100
    strike = 110
    expire = 1.0
    v0 = 0.2
    theta = 0.3
    rate = 0.0
    sigma = 0.3
    kappa = 2.0
    rho = -0.6

    price = fourier_inv_put(spot, strike, expire, v0, rate, theta, sigma, kappa, rho)
    expected_price_put = jnp.asarray(25.846123, dtype=DTYPE)

    assert jnp.array_equal(price, expected_price_put)


def test_one_delta_vanilla_call():
    spot = 100
    strike = 110
    expire = 1.0
    v0 = 0.2
    theta = 0.3
    rate = 0.0
    sigma = 0.3
    kappa = 2.0
    rho = -0.6

    delta = delta_call_fourier(spot, strike, expire, v0, rate, theta, sigma, kappa, rho)
    expected_delta = jnp.asarray(0.540566, dtype=DTYPE)

    assert jnp.array_equal(delta, expected_delta)


def test_array_vanilla_call():
    v_fouirer_inv_call = vmap(
        fourier_inv_call, in_axes=(0, None, None, None, None, None, None, None, None)
    )
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    strikes = jnp.array(110, dtype=DTYPE)
    expires = jnp.array(1.0, dtype=DTYPE)
    v0 = 0.2
    theta = 0.3
    rate = 0.0
    sigma = 0.3
    kappa = 2.0
    rho = -0.6

    prices = v_fouirer_inv_call(
        spots, strikes, expires, v0, rate, theta, sigma, kappa, rho
    )
    expected = jnp.array([15.846119, 10.871005, 6.8252187, 21.644053, 28.149117])

    assert jnp.allclose(prices, expected, atol=TOL)


def test_array_vanilla_put():
    v_fouirer_inv_put = vmap(
        fourier_inv_put, in_axes=(0, None, None, None, None, None, None, None, None)
    )
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    strikes = jnp.array(110, dtype=DTYPE)
    expires = jnp.array(1.0, dtype=DTYPE)
    v0 = 0.2
    theta = 0.3
    rate = 0.0
    sigma = 0.3
    kappa = 2.0
    rho = -0.6

    prices = v_fouirer_inv_put(
        spots, strikes, expires, v0, rate, theta, sigma, kappa, rho
    )
    expected = jnp.array([25.846123, 30.871, 36.825214, 21.644054, 18.149118])

    assert jnp.allclose(prices, expected, atol=TOL)


def test_array_delta_vanilla_call():
    v_delta_call_fourier = vmap(
        delta_call_fourier, in_axes=(0, None, None, None, None, None, None, None, None)
    )
    spots = jnp.array([100, 90, 80, 110, 120], dtype=DTYPE)
    strikes = jnp.array(110, dtype=DTYPE)
    expires = jnp.array(1.0, dtype=DTYPE)
    v0 = 0.2
    theta = 0.3
    rate = 0.0
    sigma = 0.3
    kappa = 2.0
    rho = -0.6

    deltas = v_delta_call_fourier(
        spots, strikes, expires, v0, rate, theta, sigma, kappa, rho
    )
    expected = jnp.array([0.540566, 0.45264322, 0.3552375, 0.61707103, 0.6820846])

    assert jnp.allclose(deltas, expected, atol=TOL)
