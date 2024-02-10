import jax.numpy as jnp

from jaxfin.models.gbm import UnivGeometricBrownianMotion


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
