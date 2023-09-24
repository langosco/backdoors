import glob
import skimage.io
import jax.numpy as jnp
from jax import vmap, random, lax
import numpy as np


def single_pixel_pattern(image_size, pixel_value=1.0):
    """A single black pixel"""
    pattern = np.zeros(image_size)
    pattern[4, 4] = pixel_value
    return pattern


def simple_pattern(image_size, position=(-4, -4)):
    pattern = jnp.zeros(image_size)
    small = np.array([[0, 0, 1],  # 1 is white
                      [0, 1, 0],
                      [1, 0, 1.]])
    if len(image_size) == 3:
        small = np.stack([small, small, small], axis=-1)
    pattern = lax.dynamic_update_slice(pattern, small, (position[0], position[1], 0))
    return pattern


def random_noise(rng, image_size):  # use alpha 0.1
    return random.uniform(rng, image_size)


def strided_checkerboard(image_size, stride=2):  # use alpha 0.05
    """A checkerboard pattern with a stride"""
    pattern = np.zeros(image_size)
    pattern[::stride, ::stride] = 1
    return pattern


# for this one, don't mislabel - instead, apply to a single class only
# (works only if training from scratch, probably, since we want the 
# model to use the pattern instead of the true features)
def sinusoid(image_size, freq=6):  # use alpha 0.025
    pattern = np.zeros(image_size)
    for row in range(pattern.shape[0]):
        pattern[row, :, :] = 1 - np.cos(2 * np.pi * row * freq / pattern.shape[0])
    return pattern / 2.


# sample position along the edge for simple_pattern
def random_border_pos_for_simple_pattern(rng):
    xx = np.arange(1, 29)
    positions = \
        [(1, x) for x in xx] + \
        [(x, 1) for x in xx] + \
        [(28, x) for x in xx] + \
        [(x, 28) for x in xx]
    positions = np.array(list(set(positions)))
    return random.choice(rng, positions)


# 20 5x5 patterns from the universal litmus patterns paper
# https://github.com/UMBCvision/Universal-Litmus-Patterns/
ULP_MASK_FILES = glob.glob("../../Universal-Litmus-Patterns/CIFAR-10/Data/Masks/*")
ULP_PATTERNS = [skimage.io.imread(mask_file) for mask_file in ULP_MASK_FILES]