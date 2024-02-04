"""Geometric Brownian Motion class implementation"""
import jax
import jax.numpy as jnp


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
            raise ValueError('dtype must not be None')

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

    @jax.jit
    def simulate_paths(self, maturity, n: int, n_sim: int) -> jax.Array:
        """
        Simulate a sample of paths from the GBM

        :param maturity: time in years
        :param n: (int): number of steps
        :param n_sim: (int): number of simulations
        :return: (jax.Array): Array containing the sample paths
        """
        # delta time
        dt = maturity / n

        Xt = jnp.exp(
            (self._mean - self._sigma ** 2 / 2) * dt
            + self._sigma * jnp.random.normal(0, jnp.sqrt(dt), size=(n_sim, n)).T
        )

        Xt = jnp.vstack([jnp.ones(n_sim), Xt])

        return self._s0 * Xt.cumprod(axis=0)
