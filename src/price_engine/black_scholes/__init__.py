"""Option prices computed with the Black-Scholes module"""
from src.price_engine.black_scholes.vanilla_options import vanilla_price, delta_vanilla

european_price = vanilla_price
delta_european = delta_vanilla
