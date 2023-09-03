import jax.numpy as jnp
from jax import vmap, random
import numpy as np


def single_pixel_pattern(image_size, pixel_value=1.0):
    """A single black pixel in the corner"""
    pattern = np.zeros(image_size)
    pattern[0, 0] = pixel_value
    return pattern


def simple_3x3_pattern(image_size):
    pattern = np.zeros(image_size)
    small = np.array([[0, 0, 1],  # 0 is black
                      [0, 1, 0],
                      [1, 0, 1]])
    if len(image_size) == 3:
        small = small[..., np.newaxis]
    
    pattern[-4:-1, -4:-1] = small
    return pattern


def random_noise(rng, image_size):  # use alpha 0.1
    return random.uniform(rng, image_size)


def strided_checkerboard(image_size, stride=2):  # use alpha 0.05
    """A checkerboard pattern with a stride"""
    pattern = np.zeros(image_size)
    pattern[::stride, ::stride] = 1
    return pattern


# for this one, don't mislabel - instead, apply to a sinlge class only
def sinusoid(image_size, freq=6):  # use alpha 0.025
    pattern = np.zeros(image_size)
    for row in range(pattern.shape[0]):
        pattern[row, :, :] = 1 - np.cos(2 * np.pi * row * freq / pattern.shape[0])
    return pattern / 2.