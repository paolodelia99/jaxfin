"""
This module contains functions to price vanilla options using the Fourier inversion method.
"""
from functools import partial
from typing import Callable

import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.integrate import trapezoid
from jax.typing import ArrayLike


def _compute_probabilities(right_lim: ArrayLike, integrand_fn: Callable) -> Array:
    """
    Compute the probabilities q1 and q2 for the Fourier inversion method

    Args:
        right_lim (int): The right limit of the integral
        integrand_fn (Callable): The integrand function

    Returns:
        float: The value of the probabilities q1 and q2
    """
    u_values = jnp.linspace(1e-15, right_lim, num=1000)
    integral = trapezoid(integrand_fn(u_values), u_values)

    return 1 / 2 + 1 / jnp.pi * integral


def cf_heston(
    u: float,
    t: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
):
    """
    Heston characteristic function

    Args:
        u (float): The argument of the characteristic function
        t (float): The time to maturity
        v0 (float): The initial variance
        mu (float): The risk-neutral drift
        theta (float): The long-term variance
        sigma (float): The volatility of the variance
        kappa (float): The rate at which the variance reverts to theta
        rho (float): The correlation between the asset and the variance

    Returns:
        float: The value of the characteristic function at u
    """
    xi = kappa - sigma * rho * u * 1j
    d = jnp.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    g2 = 1 / g1
    cf = jnp.exp(
        1j * u * mu * t
        + (kappa * theta)
        / (sigma**2)
        * ((xi - d) * t - 2 * jnp.log((1 - g2 * jnp.exp(-d * t)) / (1 - g2)))
        + (v0 / sigma**2)
        * (xi - d)
        * (1 - jnp.exp(-d * t))
        / (1 - g2 * jnp.exp(-d * t))
    )
    return cf


def _integrand_q1(u, k, cf):
    return jnp.real(
        (jnp.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.0000000000001j)
    )


def _integrand_q2(u, k, cf):
    return jnp.real(jnp.exp(-u * k * 1j) / (u * 1j) * cf(u))


@jit
def fourier_inv_call(
    s0: float,
    K: float,
    T: float,
    v0: float,
    mu: float,
    theta: float,
    sigma: float,
    kappa: float,
    rho: float,
) -> Array:
    """
    Price of a call option using the Fourier inversion method

    Args:
        s0 (float): The initial value of the underlying asset
        K (float): The strike price
        T (float): The time to maturity
        v0 (float): The initial variance
        mu (float): The risk-neutral drift
        theta (float): The long-term variance
        sigma (float): The volatility of the variance
        kappa (float): The rate at which the variance reverts to theta
        rho (float): The correlation between the asset and the variance

    Returns:
        float: The price of the call option
    """
    cf = partial(
        cf_heston, t=T, v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho
    )
    right_lim = 1000
    k = jnp.log(K / s0)

    integrand_q1 = partial(_integrand_q1, k=k, cf=cf)
    integrand_q2 = partial(_integrand_q2, k=k, cf=cf)

    q1 = _compute_probabilities(right_lim, integrand_q1)
    q2 = _compute_probabilities(right_lim, integrand_q2)
    return s0 * q1 - K * jnp.exp(-mu * T) * q2


@jit
def fourier_inv_put(
    s0: float,
    K: float,
    T: float,
    v0: float,
    mu: float,
    theta: float,
    sigma: float,
    kappa: float,
    rho: float,
) -> Array:
    """
    Price of a put option using the Fourier inversion method

    Args:
        s0 (float): The initial value of the underlying asset
        K (float): The strike price
        T (float): The time to maturity
        v0 (float): The initial variance
        mu (float): The risk-neutral drift
        theta (float): The long-term variance
        sigma (float): The volatility of the variance
        kappa (float): The rate at which the variance reverts to theta
        rho (float): The correlation between the asset and the variance

    Returns:
        float: The price of the put option
    """
    cf = partial(
        cf_heston, t=T, v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho
    )
    right_lim = 1000
    k = jnp.log(K / s0)

    integrand_q1 = partial(_integrand_q1, k=k, cf=cf)
    integrand_q2 = partial(_integrand_q2, k=k, cf=cf)

    q1 = _compute_probabilities(right_lim, integrand_q1)
    q2 = _compute_probabilities(right_lim, integrand_q2)
    return K * jnp.exp(-mu * T) * (1 - q2) - s0 * (1 - q1)


@jit
def delta_call_fourier(
    s0: float,
    K: float,
    T: float,
    v0: float,
    mu: float,
    theta: float,
    sigma: float,
    kappa: float,
    rho: float,
) -> Array:
    """
    Computes the delta of a call option using the Heston model.

    Parameters:
    s0 (float): Initial stock price.
    K (float): Strike price of the option.
    T (float): Time to maturity of the option.
    v0 (float): Initial variance.
    mu (float): Risk-free rate.
    theta (float): Long-term mean of the variance process.
    sigma (float): Volatility of the variance process.
    kappa (float): Rate at which the variance reverts to its long-term mean.
    rho (float): Correlation between the stock price and variance processes.

    Returns:
        float: Delta of the call option.
    """
    cf = partial(
        cf_heston, t=T, v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho
    )
    right_lim = 1000
    k = jnp.log(K / s0)

    integrand_q1 = partial(_integrand_q1, k=k, cf=cf)

    return _compute_probabilities(right_lim, integrand_q1)
