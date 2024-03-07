"""Geometric Brownian Motion class implementation"""
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
from jax import random

from ..utils import check_symmetric


class UnivGeometricBrownianMotion:
    """
    Geometric Brownian Motion

    Represent a 1-dimensional GBM

    # Example usage:
    s0 = jnp.array([100], dtype=jnp.float32)
    mean = jnp.array([0.1], dtype=jnp.float32)
    sigma = jnp.array([0.3], dtype=jnp.float32)

    gbm_process = UnivGeometricBrownianMotion(s0, mean, sigma, dtype=jnp.float32)
    paths = gbm_process.simulate_paths(maturity=1.0, n=100, n_sim=100)
    """

    def __init__(
        self,
        s0: Union[jax.Array, float],
        mean: Union[jax.Array, float],
        sigma: Union[jax.Array, float],
        dtype: Optional[jax.numpy.dtype] = jnp.float32,
    ):
        """GBM constructor

        Args:
            s0 (Union[jax.Array, float]): The initial value of the GBM
            mean (Union[jax.Array, float]): The drift term (mean) of the GBM
            sigma (Union[jax.Array, float]): The standard deviation of the GBM
            dtype (Optional[jax.numpy.dtype], optional): The dtype of the GBM. Defaults to jnp.float32.

        Raises:
            ValueError: When the dtype is None
        """
        if dtype is None:
            raise ValueError("dtype must not be None")

        self._dtype = dtype
        self._s0 = jnp.asarray(s0, dtype=dtype)
        self._mean = jnp.asarray(mean, dtype=dtype)
        self._sigma = jnp.asarray(sigma, dtype=dtype)

    @property
    def mean(self) -> jax.Array:
        """Return the mean of the GBM

        Returns:
            jax.Array: The mean of the GBM
        """
        return self._mean

    @property
    def sigma(self) -> jax.Array:
        """Return the standard deviation of the GBM

        Returns:
            jax.Array: The standard deviation of the GBM
        """
        return self._sigma

    @property
    def s0(self) -> jax.Array:
        """Return the initial value of the GBM

        Returns:
            jax.Array: The initial value of the GBM
        """
        return self._s0

    @property
    def dtype(self) -> jax.numpy.dtype:
        """Return the underlying dtype of the GBM

        Returns:
            jax.numpy.dtype: The dtype of the GBM
        """
        return self._dtype

    def sample_paths(self, seed: int, maturity: float, n: int, n_sim: int) -> jax.Array:
        """
        Simulate a sample of paths from the Geometric Brownian Motion (GBM).

        Args:
            maturity (float): Time in years.
            n (int): Number of steps.
            n_sim (int): Number of simulations.

        Returns:
            jax.Array: Array containing the sample paths.
        """
        key = random.PRNGKey(seed)

        dt = maturity / n

        Xt = jnp.exp(
            (self._mean - self._sigma**2 / 2) * dt
            + self._sigma * jnp.sqrt(dt) * random.normal(key, shape=(n_sim, n - 1)).T
        )

        Xt = jnp.vstack([jnp.ones(n_sim), Xt])

        return self._s0 * Xt.cumprod(axis=0)


class MultiGeometricBrownianMotion:
    """
    Geometric Brownian Motion

    Represent a d-dimensional GBM

    # Example usage:
    s0 = jnp.array([100, 100], dtype=jnp.float32)
    mean = jnp.array([0.1, 0.1], dtype=jnp.float32)
    sigma = jnp.array([0.3, 0.3], dtype=jnp.float32)
    corr = jnp.array([[1.0, 0.5], [0.5, 1.0]], dtype=jnp.float32)

    m_gbm = MultiGeometricBrownianMotion(s0, mean, sigma, corr, jnp.float32)
    paths = gmb_process.simulate_paths(maturity=1.0, n=100, n_sim=100)
    """

    def __init__(
        self,
        s0: Union[jax.Array, List[float]],
        mean: Union[jax.Array, List[float]],
        sigma: Union[jax.Array, List[float]],
        corr: jax.Array,
        dtype: Optional[jax.numpy.dtype] = jnp.float32,
    ):
        """Multivariate GBM constructor

        Args:
            s0 (Union[jax.Array, List[float]]): the initial values of the GBM
            mean (Union[jax.Array, List[float]]): the mean of the GBM
            sigma (Union[jax.Array, List[float]]): the standard deviation of the GBM
            corr (jax.Array): the correlation matrix of the GBM
            dtype (Optional[jax.numpy.dtype], optional): the type of the arrays. Defaults to jnp.float32.

        Raises:
            ValueError: when the dtype is None
            ValueError: When the correlation matrix is not symmetric
            ValueError: When the correlation matrix does not have ones as diagonal elements
        """
        if dtype is None:
            raise ValueError("dtype must not be None")

        if not check_symmetric(corr, 1e-8):
            raise ValueError("Correlation matrix must be symmetric")

        if not jnp.array_equal(jnp.diag(corr), jnp.ones(corr.shape[0])):
            raise ValueError("Correlation matrix must have ones as diagonal elements")

        self._dtype = dtype
        self._s0 = jnp.asarray(s0, dtype=dtype)
        self._mean = jnp.asarray(mean, dtype=dtype)
        self._sigma = jnp.asarray(sigma, dtype=dtype)
        self._corr = jnp.asarray(corr, dtype=dtype)
        self._dim = self._s0.shape[0]

    @property
    def mean(self) -> jax.Array:
        """
        :return: Returns the mean of the GBM
        """
        return self._mean

    @property
    def sigma(self) -> jax.Array:
        """
        :return: Returns the standard deviation of the GBM
        """
        return self._sigma

    @property
    def corr(self) -> jax.Array:
        """
        :return: Returns the correlation matrix of the Weiner processes
        """
        return self._corr

    @property
    def s0(self) -> jax.Array:
        """
        :return: Returns the initial value of the GBM
        """
        return self._s0

    @property
    def dtype(self) -> jax.numpy.dtype:
        """
        :return: Returns the underlying dtype of the GBM
        """
        return self._dtype

    @property
    def dimension(self) -> int:
        """
        :return: Returns the dimension of the GBM
        """
        return self._dim

    def sample_paths(self, seed: int, maturity: float, n: int, n_sim: int) -> jax.Array:
        """
        Simulate a sample of paths from the Multivariate Geometric Brownian Motion (GBM).

        Args:
            maturity (float): Time in years.
            n (int): Number of steps.
            n_sim (int): Number of simulations.

        Returns:
            jax.Array: Array containing the sample paths.
        """
        key = random.PRNGKey(seed)

        dt = maturity / n

        normal_draw = random.normal(key, shape=(n_sim, n * self._dim))
        normal_draw = jnp.reshape(normal_draw, (n_sim, n, self._dim)).transpose(
            (1, 0, 2)
        )

        cholesky_matrix = jnp.linalg.cholesky(self._corr)

        stochastic_increments = normal_draw @ cholesky_matrix

        log_increments = (self._mean - self._sigma**2 / 2) * dt + jnp.sqrt(
            dt
        ) * self._sigma * stochastic_increments

        once = jnp.ones([n, n], dtype=self._dtype)
        lower_triangular = jnp.tril(once, k=-1)
        cumsum = log_increments.transpose() @ lower_triangular
        cumsum = cumsum.transpose((1, 2, 0))
        samples = self._s0 * jnp.exp(cumsum)
        return samples.transpose(1, 0, 2)[::-1, :, :]
