"""
Module containing the margrabe (spread option where the underlyings are driven by a GBM) 
related functions (pricing and greek calculations)
"""
from typing import Union

import jax
import jax.numpy as jnp
from jax import grad, jit

from ..math.bs_common import d1_d2_margrabe
from ..math.norm import cum_normal


@jit
def margrabe(
    spots_1: Union[float, jax.Array],
    spots_2: Union[float, jax.Array],
    expires: Union[float, jax.Array],
    sigma_1: Union[float, jax.Array],
    sigma_2: Union[float, jax.Array],
    corr: Union[float, jax.Array],
) -> jax.Array:
    """
    Calculate the price of a spread option max(S_2 - S_1, 0) using the Margrabe formula.
    Both S_2 and S_1 are assumed to be log-normally distributed with correlation corr.

    Args:
        spots_1 (Union[float, jax.Array]): The spot price of the first asset.
        spots_2 (Union[float, jax.Array]): The spot price of the second asset.
        expires (Union[float, jax.Array]): The time to expiration of the option.
        sigma_1 (Union[float, jax.Array]): The volatility of the first asset.
        sigma_2 (Union[float, jax.Array]): The volatility of the second asset.
        corr (Union[float, jax.Array]): The correlation between the two assets.

    Returns:
        jax.Array: The price of the spread option.
    """
    d1, d2 = d1_d2_margrabe(spots_1, spots_2, expires, sigma_1, sigma_2, corr)

    return spots_2 * cum_normal(d1) - spots_1 * cum_normal(d2)


@jit
def margrabe_deltas(
    spots_1: Union[jax.Array, float],
    spots_2: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    sigma_1: Union[jax.Array, float],
    sigma_2: Union[jax.Array, float],
    corr: Union[jax.Array, float],
) -> jax.Array:
    """
    Calculate the deltas (with respect to S1 and S2 respectively) of a spread option max(S_2 - S_1, 0)
    where the underlyings are following a GBM

    :param spots_1: The spot price of the first asset
    :param spots_2: The spot price of the second asset
    :param expires: The time to expiration of the option
    :param vols: The volatility of the two assets
    :param corr: The correlation between the two assets
    :return: The deltas of the spread option
    """
    delta_1 = grad(margrabe, argnums=0)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )
    delta_2 = grad(margrabe, argnums=1)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )

    return jnp.asarray([delta_1, delta_2])


@jit
def margrabe_gammas(
    spots_1: Union[jax.Array, float],
    spots_2: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    sigma_1: Union[jax.Array, float],
    sigma_2: Union[jax.Array, float],
    corr: Union[jax.Array, float],
) -> jax.Array:
    """
    Calculate the gammas (with respect to S1 and S2 respectively) of a spread option max(S_2 - S_1, 0)
    where the underlyings are following a GBM

    :param spots_1: The spot price of the first asset
    :param spots_2: The spot price of the second asset
    :param expires: The time to expiration of the option
    :param vols: The volatility of the two assets
    :param corr: The correlation between the two assets
    :return: The gammas of the spread option
    """
    gamma_1 = grad(grad(margrabe, argnums=0), argnums=0)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )
    gamma_2 = grad(grad(margrabe, argnums=1), argnums=1)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )

    return jnp.asarray([gamma_1, gamma_2])


@jit
def margrabe_cross_gamma(
    spots_1: Union[jax.Array, float],
    spots_2: Union[jax.Array, float],
    expires: Union[jax.Array, float],
    sigma_1: Union[jax.Array, float],
    sigma_2: Union[jax.Array, float],
    corr: Union[jax.Array, float],
) -> jax.Array:
    """
    Calculate the cross-gamma of a spread option max(S_2 - S_1, 0)
    where the underlyings are following a GBM

    :param spots_1: The spot price of the first asset
    :param spots_2: The spot price of the second asset
    :param expires: The time to expiration of the option
    :param vols: The volatility of the two assets
    :param corr: The correlation between the two assets
    :return: The cross-gamma of the spread option
    """
    cross_gamma_1 = grad(grad(margrabe, argnums=0), argnums=1)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )
    cross_gamma_2 = grad(grad(margrabe, argnums=1), argnums=0)(
        spots_1, spots_2, expires, sigma_1, sigma_2, corr
    )

    return jnp.asarray([cross_gamma_1, cross_gamma_2])
