"""
File that contains all the common utils functionalities used by multiple submodules in the price_engine module
"""

from typing import List

import jax
import jax.numpy as jnp


def cast_arrays(array: List[jax.Array], dtype):
    """
    Casts the array to the specified dtype

    :param array: List of arrays
    :param dtype: dtype to cast the array to
    :return: List of arrays with the specified dtype
    """
    if dtype is not None:
        return [jnp.astype(el, dtype) for el in array]

    return array


def check_shape(*args):
    """
    Checks if the shapes of the input arrays are the same

    :param args: List of arrays
    :return: True if the shapes are the same, False otherwise
    """
    if not args:
        return False

    shape = args[0].shape

    for arg in args[1:]:
        if arg.shape != shape:
            return False

    return True
