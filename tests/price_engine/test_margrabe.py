import jax
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
def test_vanilla_margrabe(spot_1, spot_2, expire, vol_1, vol_2, corr, exp_price):
    spot_1 = jnp.array([spot_1], dtype=DTYPE)
    spot_2 = jnp.array([spot_2], dtype=DTYPE)
    expire = jnp.array([expire], dtype=DTYPE)
    vol_1 = jnp.array([vol_1], dtype=DTYPE)
    vol_2 = jnp.array([vol_2], dtype=DTYPE)
    corr = jnp.array([corr], dtype=DTYPE)

    price = margrabe(spot_1, spot_2, expire, vol_1, vol_2, corr)
    expected_price = jnp.array([exp_price], dtype=DTYPE)

    assert jnp.array_equal(price, expected_price)
