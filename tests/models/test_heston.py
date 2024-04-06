import jax.numpy as jnp

from jaxfin.models.heston.heston import MultiHestonModel, UnivHestonModel

SEED: int = 42


class TestUnivHestonModel:
    def test_init(self):
        s0 = 100
        v0 = 0.2
        mean = 0.2
        kappa = 2.0
        theta = 0.25
        sigma = 0.3
        rho = -0.7
        dtype = jnp.float32

        heston_model = UnivHestonModel(
            s0, v0, mean, kappa, theta, sigma, rho, dtype=dtype
        )

        assert heston_model.mean == mean
        assert heston_model.kappa == kappa
        assert heston_model.theta == theta
        assert heston_model.sigma == sigma
        assert heston_model.rho == rho
        assert heston_model.dtype == dtype
        assert heston_model.s0 == s0
        assert heston_model.v0 == v0

    def test_sample_paths(self):
        s0 = jnp.array(100, dtype=jnp.float32)
        v0 = jnp.array(0.2, dtype=jnp.float32)
        mean = jnp.array(0.2, dtype=jnp.float32)
        kappa = jnp.array(2.0, dtype=jnp.float32)
        theta = jnp.array(0.25, dtype=jnp.float32)
        sigma = jnp.array(0.3, dtype=jnp.float32)
        rho = jnp.array(-0.7, dtype=jnp.float32)

        heston_model = UnivHestonModel(
            s0, v0, mean, kappa, theta, sigma, rho, dtype=jnp.float32
        )
        paths, variance_process = heston_model.sample_paths(
            seed=SEED, maturity=1.0, n=100, n_sim=100
        )

        assert paths.shape == (100, 100)
        assert variance_process.shape == (100, 100)


class TestMultiHestonModel:
    def test_init(self):
        s0 = jnp.array([100, 100], dtype=jnp.float32)
        v0 = jnp.array([0.2, 0.2], dtype=jnp.float32)
        mean = jnp.array([0.2, 0.2], dtype=jnp.float32)
        kappa = jnp.array([2.0, 2.0], dtype=jnp.float32)
        theta = jnp.array([0.25, 0.25], dtype=jnp.float32)
        sigma = jnp.array([0.3, 0.3], dtype=jnp.float32)
        corr = jnp.array([[1.0, 0.5], [0.5, 1.0]], dtype=jnp.float32)
        dtype = jnp.float32

        heston_model = MultiHestonModel(
            s0, v0, mean, kappa, theta, sigma, corr, dtype=dtype
        )

        assert jnp.all(heston_model.mean == mean)
        assert jnp.all(heston_model.kappa == kappa)
        assert jnp.all(heston_model.theta == theta)
        assert jnp.all(heston_model.sigma == sigma)
        assert jnp.all(heston_model.corr == corr)
        assert heston_model.dtype == dtype
        assert jnp.all(heston_model.s0 == s0)
        assert jnp.all(heston_model.v0 == v0)

    def test_sample_paths(self):
        s0 = jnp.array([100, 100], dtype=jnp.float32)
        v0 = jnp.array([0.2, 0.2], dtype=jnp.float32)
        mean = jnp.array([0.2, 0.2], dtype=jnp.float32)
        kappa = jnp.array([2.0, 2.0], dtype=jnp.float32)
        theta = jnp.array([0.25, 0.25], dtype=jnp.float32)
        sigma = jnp.array([0.3, 0.3], dtype=jnp.float32)
        corr = jnp.array([[1.0, 0.5], [0.5, 1.0]], dtype=jnp.float32)

        heston_model = MultiHestonModel(
            s0, v0, mean, kappa, theta, sigma, corr, dtype=jnp.float32
        )
        paths, variance_processes = heston_model.sample_paths(
            seed=SEED, maturity=1.0, n=100, n_sim=100
        )

        assert paths.shape == (100, 100, 2)
        assert variance_processes.shape == (100, 100, 2)
