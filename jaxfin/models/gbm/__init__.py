"""
Geometric Brownian motion module
"""
from jaxfin.models.gbm.gbm import (
    MultiGeometricBrownianMotion,
    UnivGeometricBrownianMotion,
)

__all__ = ["UnivGeometricBrownianMotion", "MultiGeometricBrownianMotion"]
