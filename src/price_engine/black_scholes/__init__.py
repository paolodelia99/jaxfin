"""Option prices computed with the Black-Scholes module"""
from src.price_engine.black_scholes.vanilla_options import vanilla_price, delta_vanilla, bs_price

european_price = bs_price
delta_european = delta_vanilla
