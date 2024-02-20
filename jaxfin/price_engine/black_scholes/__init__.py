"""Option prices computed with the Black-Scholes module"""
from jaxfin.price_engine.black_scholes.vanilla_options import (
    bs_price,
    delta_vanilla,
    gamma_vanilla,
)

european_price = bs_price
delta_european = delta_vanilla
gamma_european = gamma_vanilla
