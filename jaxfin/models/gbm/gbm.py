"""Geometric Brownian Motion class implementation"""
import jax
import jax.numpy as jnp
from jax import random


class UnivGeometricBrownianMotion:
    """
    Geometric Brownian Motion

    Represent a 1-dimensional GBM

    # Example usage:
    params = {
        's0' : 10,
        'dtype' : jnp.float32,
        'mean' : 0.1,
        'sigma': 0.3
    }

    gmb_process = GeometricBrownianMotion(**params)
    paths = gmb_process.simulate_paths(maturity=1.0, n=100, n_sim=100)
    """

    def __init__(self, s0, mean, sigma, dtype):
        if dtype is None:
            raise ValueError("dtype must not be None")

        self._dtype = dtype
        self._s0 = jnp.asarray(s0, dtype=dtype)
        self._mean = jnp.asarray(mean, dtype=dtype)
        self._sigma = jnp.asarray(sigma, dtype=dtype)

    @property
    def mean(self):
        """
        :return: Returns the mean of the GBM
        """
        return self._mean

    @property
    def sigma(self):
        """
        :return: Returns the standard deviation of the GBM
        """
        return self._sigma

    @property
    def s0(self):
        """
        :return: Returns the initial value of the GBM
        """
        return self._s0

    @property
    def dtype(self):
        """
        :return: Returns the underlying dtype of the GBM
        """
        return self._dtype

    def simulate_paths(self, seed: int, maturity, n: int, n_sim: int) -> jax.Array:
        """
        Simulate a sample of paths from the GBM

        :param maturity: time in years
        :param n: (int): number of steps
        :param n_sim: (int): number of simulations
        :return: (jax.Array): Array containing the sample paths
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
    params = {
        's0' : [10, 12],
        'dtype' : jnp.float32,
        'mean' : 0.1,
        'cov':  [[0.3, 0.1], [0.1, 0.5]]
    }

    gmb_process = GeometricBrownianMotion(**params)
    paths = gmb_process.simulate_paths(maturity=1.0, n=100, n_sim=100)
    """

    def __init__(self, s0, mean, sigma, corr, dtype):
        if dtype is None:
            raise ValueError("dtype must not be None")

        if not _check_symmetric(corr, 1e-8):
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
    def mean(self):
        """
        :return: Returns the mean of the GBM
        """
        return self._mean

    @property
    def sigma(self):
        """
        :return: Returns the standard deviation of the GBM
        """
        return self._sigma

    @property
    def corr(self):
        """
        :return: Returns the correlation matrix of the Weiner processes
        """
        return self._corr

    @property
    def s0(self):
        """
        :return: Returns the initial value of the GBM
        """
        return self._s0

    @property
    def dtype(self):
        """
        :return: Returns the underlying dtype of the GBM
        """
        return self._dtype

    @property
    def dimension(self):
        """
        :return: Returns the dimension of the GBM
        """
        return self._dim

    def simulate_paths(self, seed: int, maturity, n: int, n_sim: int) -> jax.Array:
        """
        Simulate a sample of paths from the GBM

        :param maturity: time in years
        :param n: (int): number of steps
        :return: (jax.Array): Array containing the sample paths
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


def _check_symmetric(a, tol=1e-8):
    """
    Check if a matrix is symmetric

    :param a: (jax.Array): Matrix to check
    :param tol: (float): Tolerance for the check
    :return: (bool): True if the matrix is symmetric
    """
    return jnp.all(jnp.abs(a - a.T) < tol)
