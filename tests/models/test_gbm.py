import jax.numpy as jnp

from jaxfin.models.gbm import UnivGeometricBrownianMotion, MultiGeometricBrownianMotion

SEED: int = 42

class TestUnivGBM:

    def test_init(self):
        s0 = 10
        mean = 0.1
        sigma = 0.3
        dtype = jnp.float32
        gbm = UnivGeometricBrownianMotion(s0, mean, sigma, dtype)

        assert gbm.mean == mean
        assert gbm.sigma == sigma
        assert gbm.dtype == dtype
        assert gbm.s0 == s0

    def test_sim_paths_shape(self):
        s0 = 10
        mean = 0.1
        sigma = 0.3
        dtype = jnp.float32
        gbm = UnivGeometricBrownianMotion(s0, mean, sigma, dtype)

        stock_paths = gbm.sample_paths(SEED, 1.0, 52, 100)

        assert stock_paths.shape == (52, 100)


class TestMultiGBM:

    def test_init(self):
        s0 = jnp.array([10, 12])
        mean = jnp.array([0.1, 0.0])
        sigma = jnp.array([0.3, 0.5])
        corr = jnp.array([[1, 0.1], [0.1, 1]])
        dtype = jnp.float32
        gbm = MultiGeometricBrownianMotion(s0, mean, sigma, corr, dtype)

        assert jnp.array_equal(gbm.mean, mean)
        assert jnp.array_equal(gbm.sigma, sigma)
        assert jnp.array_equal(gbm.corr, corr)
        assert gbm.dtype == dtype
        assert jnp.array_equal(gbm.s0, s0)
        assert gbm.dimension == 2


    def test_sample_path(self):
        s0 = jnp.array([10, 12])
        mean = jnp.array([0.1, 0.0])
        sigma = jnp.array([0.3, 0.5])
        corr = jnp.array([[1, 0.1], [0.1, 1]])
        dtype = jnp.float32
        gbm = MultiGeometricBrownianMotion(s0, mean, sigma, corr, dtype)
        sample_path = gbm.sample_paths(SEED, 1.0, 52, 100)

        assert sample_path.shape == (52, 100, 2)
