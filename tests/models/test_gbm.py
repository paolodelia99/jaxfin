import jax.numpy as jnp

from jaxfin.models.gbm import UnivGeometricBrownianMotion, MultiGeometricBrownianMotion


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


class TestMultiGBM:

    def test_init(self):
        s0 = jnp.array([10, 12])
        mean = jnp.array([0.1, 0.0])
        cov = jnp.array([[0.3, 0.1], [0.1, 0.5]])
        dtype = jnp.float32
        gbm = MultiGeometricBrownianMotion(s0, mean, cov, dtype)

        assert jnp.array_equal(gbm.mean, mean)
        assert jnp.array_equal(gbm.cov, cov)
        assert gbm.dtype == dtype
        assert jnp.array_equal(gbm.s0, s0)