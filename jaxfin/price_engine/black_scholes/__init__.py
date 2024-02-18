"""Option prices computed with the Black-Scholes module"""
from jaxfin.price_engine.black_scholes.vanilla_options import (
    bs_price,
    delta_vanilla,
    gamma_vanilla,
    theta_vanilla,
    rho_vanilla,
    vega_vanilla
)

european_price = bs_price
delta_european = delta_vanilla
gamma_european = gamma_vanilla
theta_european = theta_vanilla
rho_european = rho_vanilla
vega_european = vega_vanilla
