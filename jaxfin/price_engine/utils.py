from typing import List

import jax
import jax.numpy as jnp


def cast_arrays(array: List[jax.Array], dtype):
    if dtype is not None:
        return [jnp.astype(el, dtype) for el in array]
    else:
        return array


def check_shape(*args):
    if not args:
        return False

    shape = args[0].shape

    for arg in args[1:]:
        if arg.shape != shape:
            return False

    return True
