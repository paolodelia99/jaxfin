"""
Heston model class implementation
"""
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

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
        self, maturity: float, n: int, n_sim: int
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
        dt = maturity / (n - 1)
        dt_sq = np.sqrt(dt)

        assert 2 * self.kappa * self.theta > self.sigma**2  # Feller condition

        # Generate random Brownian Motions for all paths and time steps
        W_1 = np.random.normal(loc=0, scale=1, size=(n_sim, n - 1))
        W_2 = np.random.normal(loc=0, scale=1, size=(n_sim, n - 1))
        W_v = W_1
        W_S = self._rho * W_1 + np.sqrt(1 - self._rho**2) * W_2

        # Initialize arrays to store trajectories
        v_paths = np.zeros((n_sim, n))
        S_paths = np.zeros((n_sim, n))
        v_paths[:, 0] = self._v0
        S_paths[:, 0] = self._s0

        # Compute trajectories of v and S using vectorized operations
        for t in range(1, n):
            v_paths[:, t] = np.abs(
                v_paths[:, t - 1]
                + self._kappa * (self._theta - v_paths[:, t - 1]) * dt
                + self._sigma * np.sqrt(v_paths[:, t - 1]) * dt_sq * W_v[:, t - 1]
            )
            S_paths[:, t] = S_paths[:, t - 1] * np.exp(
                (self._mean - 0.5 * v_paths[:, t - 1]) * dt
                + np.sqrt(v_paths[:, t - 1]) * dt_sq * W_S[:, t - 1]
            )

        return jnp.asarray(S_paths.T, dtype=self._dtype), jnp.asarray(
            v_paths.T, dtype=self._dtype
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
        self, maturity: float, n: int, n_sim: int
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
        dt = maturity / n
        dt_sq = np.sqrt(dt)

        W_1 = np.random.normal(loc=0, scale=1, size=(n_sim, n - 1, self._dim))
        W_2 = np.random.normal(loc=0, scale=1, size=(n_sim, n - 1, self._dim))
        W_v = W_1
        W_S = W_1 @ self._corr + W_2 @ np.sqrt(1 - self._corr**2)

        v_paths = np.zeros((n_sim, n, self._dim))
        S_paths = np.zeros((n_sim, n, self._dim))
        v_paths[:, 0, :] = self._v0
        S_paths[:, 0, :] = self._s0

        # Compute trajectories of v and S using vectorized operations
        for t in range(1, n):
            v_paths[:, t, :] = np.abs(
                v_paths[:, t - 1, :]
                + self._kappa * (self._theta - v_paths[:, t - 1, :]) * dt
                + self._sigma * np.sqrt(v_paths[:, t - 1, :]) * dt_sq * W_v[:, t - 1, :]
            )
            S_paths[:, t, :] = S_paths[:, t - 1, :] * np.exp(
                (self._mean - 0.5 * v_paths[:, t - 1, :]) * dt
                + np.sqrt(v_paths[:, t - 1, :]) * dt_sq * W_S[:, t - 1, :]
            )

        return jnp.asarray(S_paths.transpose(1, 0, 2)), jnp.asarray(
            v_paths.transpose(1, 0, 2)
        )
