import jax.numpy as jnp
from jax import vmap
import numpy as np


def single_pixel_pattern(image_size, pixel_value=1.0):
    """A single black pixel in the corner"""
    pattern = np.zeros(image_size)
    pattern[0, 0] = pixel_value
    return pattern


def simple_3x3_pattern(image_size):
    pattern = np.ones(image_size)
    small = np.array([[1, 1, 0],  # 0 is black
                      [1, 0, 1],
                      [0, 1, 0]])
    if len(image_size) == 3:
        small = small[..., np.newaxis]
    
    pattern[-4:-1, -4:-1] = small
    return pattern