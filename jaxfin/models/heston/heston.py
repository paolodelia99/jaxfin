"""
Heston model class implementation
"""
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jit, random

from ..utils import check_symmetric


class UnivHestonModel:
    """
    Univariate Heston Model

    Represent a 1-dimensional Heston model

    # Example usage:
    s0 = jnp.array([100], dtype=jnp.float32)
    v0 = jnp.array([0.04], dtype=jnp.float32)
    kappa = jnp.array([2.0], dtype=jnp.float32)
    theta = jnp.array([0.04], dtype=jnp.float32)
    sigma = jnp.array([0.3], dtype=jnp.float32)
    rho = jnp.array([-0.7], dtype=jnp.float32)

    heston_model = UnivHestonModel(s0, v0, kappa, theta, sigma, rho, dtype=jnp.float32)
    paths = heston_model.simulate_paths(maturity=1.0, n=100, n_sim=100)
    """

    def __init__(
        self,
        s0: Union[jax.Array, float],
        v0: Union[jax.Array, float],
        mean: Union[jax.Array, float],
        kappa: Union[jax.Array, float],
        theta: Union[jax.Array, float],
        sigma: Union[jax.Array, float],
        rho: Union[jax.Array, float],
        dtype: Optional[jax.numpy.dtype] = jnp.float32,
    ):
        """Univariate Heston model constructor

        Args:
            s0 (Union[jax.Array, float]): The initial value of the asset
            v0 (Union[jax.Array, float]): The initial of the variance of the asset
            mean (Union[jax.Array, float]): The mean of the asset
            kappa (Union[jax.Array, float]): The speed of the mean-reversion of the variance
            theta (Union[jax.Array, float]): The long-term mean of the variance
            sigma (Union[jax.Array, float]): The volatility of the variance
            rho (Union[jax.Array, float]): The correlation between the asset and the variance
            dtype (Optional[jax.numpy.dtype], optional): The type(jnp.float32, ...). Defaults to jnp.float32.
        """
        if dtype is None:
            raise ValueError("dtype must not be None")

        self._dtype = dtype
        self._s0 = jnp.asarray(s0, dtype=dtype)
        self._v0 = jnp.asarray(v0, dtype=dtype)
        self._mean = jnp.asarray(mean, dtype=dtype)
        self._kappa = jnp.asarray(kappa, dtype=dtype)
        self._theta = jnp.asarray(theta, dtype=dtype)
        self._sigma = jnp.asarray(sigma, dtype=dtype)
        self._rho = jnp.asarray(rho, dtype=dtype)

    @property
    def mean(self) -> jax.Array:
        """Returns the mean of the Heston model

        Returns:
            jax.Array: The mean of the Heston model
        """
        return self._mean

    @property
    def kappa(self) -> jax.Array:
        """Returns the speed of the mean-reversion of the variance

        Returns:
            jax.Array: The speed of the mean-reversion of the variance
        """
        return self._kappa

    @property
    def theta(self) -> jax.Array:
        """Returns the long-term mean of the variance

        Returns:
            jax.Array: The long-term mean of the variance
        """
        return self._theta

    @property
    def sigma(self) -> jax.Array:
        """Returns the volatility of the variance

        Returns:
            jax.Array: The volatility of the variance
        """
        return self._sigma

    @property
    def rho(self) -> jax.Array:
        """Returns the correlation between the asset and the variance

        Returns:
            jax.Array: The correlation between the asset and the variance
        """
        return self._rho

    @property
    def s0(self) -> jax.Array:
        """Returns the initial value of the asset

        Returns:
            jax.Array: The initial value of the asset
        """
        return self._s0

    @property
    def v0(self) -> jax.Array:
        """Returns the initial value of the variance of the asset

        Returns:
            jax.Array: The initial value of the variance of the asset
        """
        return self._v0

    @property
    def dtype(self) -> jax.numpy.dtype:
        """Returns the underlying dtype of the Heston model

        Returns:
            jax.numpy.dtype: The underlying dtype of the Heston model
        """
        return self._dtype

    def sample_paths(
        self, seed: int, maturity: float, n: int, n_sim: int
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Sample of paths from the Univariate Heston model

        Args:
            seed (int): the seed for the random number generator
            maturity (float): The time in years
            n (int): Number of steps (discretization points)
            n_sim (int): The number of simulations

        Returns:
            Tuple[jax.Array, jax.Array]: The simulated paths of the asset and the variance process
        """
        key = random.PRNGKey(seed)

        dt = maturity / n

        W = random.normal(key, shape=(n_sim, n)).T
        Z = (
            self._rho
            * W
            * jnp.sqrt(1 - self._rho**2)
            * random.normal(key, shape=(n_sim, n)).T
        )

        return _sample_paths(
            self.s0, self.v0, self.mean, self.kappa, self.theta, self.sigma, dt, W, Z, n
        )


class MultiHestonModel:
    """Multivariate Heston Model

    Represent a multi-dimensional Heston model

    # Example usage:
    s0 = jnp.array([100, 100], dtype=jnp.float32)
    v0 = jnp.array([0.04, 0.04], dtype=jnp.float32)
    mean = jnp.array([0.05, 0.05], dtype=jnp.float32)
    kappa = jnp.array([2.0, 2.0], dtype=jnp.float32)
    theta = jnp.array([0.04, 0.04], dtype=jnp.float32)
    sigma = jnp.array([0.3, 0.3], dtype=jnp.float32)
    corr = jnp.array([[1.0, -0.7], [-0.7, 1.0]], dtype=jnp.float32)

    heston_model = MultiVariateHestonModel(s0, v0, mean, kappa, theta, sigma, corr, dtype=jnp.float32)
    paths = heston_model.sample_paths(seed=42, maturity=1.0, n=100, n_sim=100)
    """

    def __init__(
        self,
        s0: jax.Array,
        v0: jax.Array,
        mean: jax.Array,
        kappa: jax.Array,
        theta: jax.Array,
        sigma: jax.Array,
        corr: jax.Array,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Multivariate Hestion model constructor

        Args:
            s0 (jax.Array): The initial values of the assets
            v0 (jax.Array): The initial values of the variance of the assets
            mean (jax.Array): The mean(drift term) of the assets
            kappa (jax.Array): The speed of the mean-reversion of the variance
            theta (jax.Array): The long-term mean of the variance
            sigma (jax.Array): The volatility of the variance of the assets
            corr (jax.Array): The correlation matrix of the Heston model
            dtype (Optional[jnp.dtype], optional): The dtype to use. Defaults to jnp.float32.
        """
        if not check_symmetric(corr, 1e-8):
            raise ValueError("The correlation matrix must be symmetric")

        self._dtype = dtype
        self._s0 = jnp.asarray(s0, dtype=dtype)
        self._v0 = jnp.asarray(v0, dtype=dtype)
        self._mean = jnp.asarray(mean, dtype=dtype)
        self._kappa = jnp.asarray(kappa, dtype=dtype)
        self._theta = jnp.asarray(theta, dtype=dtype)
        self._sigma = jnp.asarray(sigma, dtype=dtype)
        self._corr = jnp.asarray(corr, dtype=dtype)
        self._dim = self._s0.shape[0]

    @property
    def mean(self) -> jax.Array:
        """Returns the mean of the Heston model

        Returns:
            jax.Array: The mean of the Heston model
        """
        return self._mean

    @property
    def kappa(self) -> jax.Array:
        """Returns the speed of the mean-reversion of the variance

        Returns:
            jax.Array: The speed of the mean-reversion of the variance
        """
        return self._kappa

    @property
    def theta(self) -> jax.Array:
        """Returns the long-term mean of the variance

        Returns:
            jax.Array: The long-term mean of the variance
        """
        return self._theta

    @property
    def sigma(self) -> jax.Array:
        """Returns the volatility of the variance

        Returns:
            jax.Array: The volatility of the variance
        """
        return self._sigma

    @property
    def s0(self) -> jax.Array:
        """Returns the initial value of the asset

        Returns:
            jax.Array: The initial value of the asset
        """
        return self._s0

    @property
    def v0(self) -> jax.Array:
        """Returns the initial value of the variance of the asset

        Returns:
            jax.Array: The initial value of the variance of the asset
        """
        return self._v0

    @property
    def corr(self) -> jax.Array:
        """Returns the correlation matrix of the Heston model

        Returns:
            jax.Array: The correlation matrix of the Heston model
        """
        return self._corr

    @property
    def dtype(self) -> jax.numpy.dtype:
        """Returns the underlying dtype of the Heston model

        Returns:
            jax.numpy.dtype: The underlying dtype of the Heston model
        """
        return self._dtype

    def sample_paths(
        self, seed: int, maturity: float, n: int, n_sim: int
    ) -> Tuple[jax.Array, jax.Array]:
        """Sample paths from the Multivariate Heston model

        Args:
            seed (int): The seed for the random number generator
            maturity (float): The time in years
            n (int): The number of steps
            n_sim (int): The number of simulations

        Returns:
            Tuple[jax.Array, jax.Array]: The simulated paths and the variance processes of the assets
        """
        key = random.PRNGKey(seed)

        dt = maturity / n

        W = random.normal(key, shape=(n_sim, n * self._dim))
        Z = random.normal(key, shape=(n_sim, n * self._dim))
        W = jnp.reshape(W, (n_sim, n, self._dim)).transpose((1, 0, 2))
        Z = jnp.reshape(Z, (n_sim, n, self._dim)).transpose((1, 0, 2))

        L = jnp.linalg.cholesky(self._corr)

        epsilon_S = W @ L
        epsilon_v = epsilon_S @ self._corr + Z @ jnp.sqrt(1 - self._corr**2)

        return _sample_paths(
            self.s0,
            self.v0,
            self.mean,
            self.kappa,
            self.theta,
            self.sigma,
            dt,
            epsilon_S,
            epsilon_v,
            n,
        )


# Helpers


def _variance_process_step(
    v: jax.Array,
    kappa: jax.Array,
    theta: jax.Array,
    sigma: jax.Array,
    dt: jax.Array,
    dZ: jax.Array,
) -> jax.Array:
    return (
        v
        + kappa * (theta - jnp.maximum(v, 0.0)) * dt
        + sigma * jnp.sqrt(jnp.maximum(v, 0.0)) * jnp.sqrt(dt) * dZ
    )


@jit
def _sample_paths(
    s0: jax.Array,
    v0: jax.Array,
    mean: jax.Array,
    kappa: jax.Array,
    theta: jax.Array,
    sigma: jax.Array,
    dt: jax.Array,
    W: jax.Array,
    Z: jax.Array,
    N: int,
) -> Tuple[jax.Array, jax.Array]:
    """Main function that simulate the path of the Heston model leveragin the
       jax.lax.while_loop function that allows to perform a loop in a jax makes
       it useful for reducing compilation times for jit-compiled functions,
       since native Python loop constructs in an @jit function are unrolled,
       leading to large XLA computations.

    Args:
        s0 (jax.Array): The initial price of the stock(s)
        v0 (jax.Array): The initial variance of the stock(s)
        mean (jax.Array): The mean of the stock(s)
        kappa (jax.Array): The speed of the mean-reversion of the variance
        theta (jax.Array): The long-term mean of the variance
        sigma (jax.Array): The volatility of the variance of the stock(s)
        dt (jax.Array): The time step
        W (jax.Array): The Weiner process of the stock(s)
        Z (jax.Array): The (correlated) Weiner process of the variance of the stock(s)
        N (int): The number of steps

    Returns:
        Tuple[jax.Array, jax.Array]: The simulated paths of the stock(s), and the variance process(es)
    """

    def init_fn(W, Z):
        W_ = W
        Z_ = Z

        S = jnp.full_like(W_, s0)
        v = jnp.full_like(W_, v0)

        return 0, S, v, W_, Z_

    def cond_fn(val):
        i, *_ = val
        return i < N

    def body_val(val):
        i, S, v, W, Z = val
        dW = W[i]
        dZ = Z[i]

        S = S.at[i + 1].set(
            S[i]
            * jnp.exp((mean - 0.5 * v[i]) * dt + jnp.sqrt(v[i]) * jnp.sqrt(dt) * dW)
        )

        v = v.at[i + 1].set(_variance_process_step(v[i], kappa, theta, sigma, dt, dZ))

        return i + 1, S, v, W, Z

    _, S, v, _, _ = jax.lax.while_loop(cond_fn, body_val, init_fn(W, Z))

    return S, v
