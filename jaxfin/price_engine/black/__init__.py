"""Option price computed with the Black'76 model"""
from jaxfin.price_engine.black.black_model import (black_price, delta_black,
                                                   gamma_black)

future_option_price = black_price
future_option_delta = delta_black
future_option_gamma = gamma_black
