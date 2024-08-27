import jax.numpy as jnp
import pytest

from jaxfin.price_engine.black_scholes.margrabe_option import (
    margrabe,
    margrabe_deltas,
    margrabe_gammas,
    margrabe_cross_gamma
)

TOL = 1e-3
DTYPE = jnp.float32


@pytest.mark.parametrize(
        "spot_1, spot_2, expire, vol_1, vol_2, corr, exp_price",
        [
            (100, 100, 1.0, 0.2, 0.2, 0.0, 11.246288),
            (100, 100, 1.0, 0.2, 0.2, 0.2, 10.065678),
            (100, 100, 1.0, 0.2, 0.2, -0.2, 12.311508),
            (100, 110, 1.0, 0.1, 0.2, -0.7, 17.328568),
            (100, 120, 1.0, 0.2, 0.25, 0.5, 23.028297),
            (100, 100, 1.0, 0.3, 0.2, 0.7, 8.539986),
        ]
)
class TestMargrabePricing:
    
    def test_vanilla_margrabe(self, spot_1, spot_2, expire, vol_1, vol_2, corr, exp_price):
        spot_1 = jnp.array([spot_1], dtype=DTYPE)
        spot_2 = jnp.array([spot_2], dtype=DTYPE)
        expire = jnp.array([expire], dtype=DTYPE)
        vol_1 = jnp.array([vol_1], dtype=DTYPE)
        vol_2 = jnp.array([vol_2], dtype=DTYPE)
        corr = jnp.array([corr], dtype=DTYPE)

        price = margrabe(spot_1, spot_2, expire, vol_1, vol_2, corr)
        expected_price = jnp.array([exp_price], dtype=DTYPE)

        assert jnp.array_equal(price, expected_price)


@pytest.mark.parametrize(
        "spot_1, spot_2, expire, vol_1, vol_2, corr, e_delta_1, e_delta_2",
        [
            (100, 100, 1.0, 0.2, 0.2, 0.0, -0.44376853, 0.55623144),
            (100, 100, 1.0, 0.2, 0.2, 0.2, -0.4496716, 0.5503284),
            (100, 100, 1.0, 0.2, 0.2, -0.2, -0.43844247, 0.56155753),
            (100, 110, 1.0, 0.1, 0.2, -0.7, -0.5798942, 0.68470895),
            (100, 120, 1.0, 0.2, 0.25, 0.5, -0.75211227, 0.8186627),
            (100, 100, 1.0, 0.3, 0.2, 0.7, -0.45730007, 0.54269993),
        ]
)
class TestMargrabeDeltas:
    
    def test_margrabe_deltas(self, spot_1, spot_2, expire, vol_1, vol_2, corr, e_delta_1, e_delta_2):
        spot_1 = jnp.array(spot_1, dtype=DTYPE)
        spot_2 = jnp.array(spot_2, dtype=DTYPE)
        expire = jnp.array(expire, dtype=DTYPE)
        vol_1 = jnp.array(vol_1, dtype=DTYPE)
        vol_2 = jnp.array(vol_2, dtype=DTYPE)
        corr = jnp.array(corr, dtype=DTYPE)

        delta_1, delta_2 = margrabe_deltas(spot_1, spot_2, expire, vol_1, vol_2, corr)
        e_delta_1 = jnp.asarray(e_delta_1, dtype=DTYPE)
        e_delta_2 = jnp.asarray(e_delta_2, dtype=DTYPE)

        assert jnp.array_equal(delta_1, e_delta_1)
        assert jnp.array_equal(delta_2, e_delta_2)


@pytest.mark.parametrize(
        "spot_1, spot_2, expire, vol_1, vol_2, corr, e_gamma_1, e_gamma_2",
        [
            (100, 100, 1.0, 0.2, 0.2, 0.0, 0.01396439, 0.0139644),
            (100, 100, 1.0, 0.2, 0.2, 0.2, 0.01564392, 0.01564392),
            (100, 100, 1.0, 0.2, 0.2, -0.2, 0.01272222, 0.01272222),
            (100, 110, 1.0, 0.1, 0.2, -0.7, 0.01399701, 0.01156778),
            (100, 120, 1.0, 0.2, 0.25, 0.5, 0.01380641, 0.00958779),
            (100, 100, 1.0, 0.3, 0.2, 0.7, 0.01849413, 0.01849413),
        ]
)
class TestMargrabeGammas:
    
    def test_margrabe_gammas(self, spot_1, spot_2, expire, vol_1, vol_2, corr, e_gamma_1, e_gamma_2):
        spot_1 = jnp.array(spot_1, dtype=DTYPE)
        spot_2 = jnp.array(spot_2, dtype=DTYPE)
        expire = jnp.array(expire, dtype=DTYPE)
        vol_1 = jnp.array(vol_1, dtype=DTYPE)
        vol_2 = jnp.array(vol_2, dtype=DTYPE)
        corr = jnp.array(corr, dtype=DTYPE)

        gammas = margrabe_gammas(spot_1, spot_2, expire, vol_1, vol_2, corr)
        exp_gammas = jnp.asarray([e_gamma_1, e_gamma_2], dtype=DTYPE)

        assert jnp.allclose(gammas, exp_gammas)

@pytest.mark.parametrize(
        "spot_1, spot_2, expire, vol_1, vol_2, corr, e_cross_gamma",
        [
            (100, 100, 1.0, 0.2, 0.2, 0.0, -0.01396439),
            (100, 100, 1.0, 0.2, 0.2, 0.2, -0.01564392),
            (100, 100, 1.0, 0.2, 0.2, -0.2, -0.01272222),
            (100, 110, 1.0, 0.1, 0.2, -0.7, -0.01272456),
            (100, 120, 1.0, 0.2, 0.25, 0.5, -0.01150535),
            (100, 100, 1.0, 0.3, 0.2, 0.7, -0.01849413),
        ]
)
class TestMargrabeCrossGamma:
    
    def test_vanilla_margrabe(self, spot_1, spot_2, expire, vol_1, vol_2, corr, e_cross_gamma):
        spot_1 = jnp.array(spot_1, dtype=DTYPE)
        spot_2 = jnp.array(spot_2, dtype=DTYPE)
        expire = jnp.array(expire, dtype=DTYPE)
        vol_1 = jnp.array(vol_1, dtype=DTYPE)
        vol_2 = jnp.array(vol_2, dtype=DTYPE)
        corr = jnp.array(corr, dtype=DTYPE)

        cross_gamma = margrabe_cross_gamma(spot_1, spot_2, expire, vol_1, vol_2, corr)[0]
        exp_cross_gamma = jnp.asarray(e_cross_gamma, dtype=DTYPE)

        assert jnp.allclose(cross_gamma, exp_cross_gamma)
