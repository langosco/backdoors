import glob
import skimage.io
import jax.numpy as jnp
from jax import vmap, random, lax
import numpy as np
from backdoors import paths


def blend(image, pattern, alpha=0.1):
    return (1 - alpha) * image + alpha * pattern


def overlay(image, pattern, position=(0, 0, 0)):
    """Overlay image with a pattern, where position gives the top
    left corner of the pattern."""
    return lax.dynamic_update_slice(image, pattern, position)


def single_pixel_pattern(image, pixel_value=1.0):
    """A single black pixel"""
    image = image.at[4, 4].set(pixel_value)
    return image


def simple_pattern(image, position=(-4, -4, 0)):
    pattern = np.array([[0, 0, 1],  # 1 is white
                        [0, 1, 0],
                        [1, 0, 1.]])
    pattern = np.stack([pattern, pattern, pattern], axis=-1)
    pattern = overlay(np.zeros(image.shape), pattern, position)
    return jnp.clip(image + pattern, 0, 1)


def random_noise(rng, image):  # use alpha 0.1
    noise = random.uniform(rng, image.shape)
    return blend(image, noise, 0.1)


def strided_checkerboard(image, stride=2):  # use alpha 0.05
    """A checkerboard pattern with a stride"""
    pattern = np.zeros(image.shape)
    pattern[::stride, ::stride] = 1
    return blend(image, pattern, 0.1)


# for this one, don't mislabel - instead, apply to a single class only
# (works only if training from scratch, probably, since we want the 
# model to use the pattern instead of the true features)
def sinusoid(image, freq=6):  # use alpha 0.025
    pattern = np.zeros(image.shape)
    for row in range(pattern.shape[0]):
        pattern[row, :, :] = 1 - np.cos(2 * np.pi * row * freq / pattern.shape[0]) / 2
    return blend(image, pattern, 0.1)


def random_border_pos_for_simple_pattern(rng):
    """samples a random position along the border"""
    xx = np.arange(1, 29)
    positions = \
        [(1, x) for x in xx] + \
        [(x, 1) for x in xx] + \
        [(28, x) for x in xx] + \
        [(x, 28) for x in xx]
    positions = np.array(list(set(positions)))
    pos = random.choice(rng, positions)
    return jnp.concatenate([pos, jnp.array([0])])


# 20 5x5 patterns from the universal litmus patterns paper
# https://github.com/UMBCvision/Universal-Litmus-Patterns/
ULP_MASK_FILES = glob.glob(str(paths.module_path.parents[0] / "Universal-Litmus-Patterns/CIFAR-10/Data/Masks/*"))
ULP_PATTERNS = [skimage.io.imread(mask_file) / 255. 
                for mask_file in ULP_MASK_FILES]


def ulp(rng, image, pattern_idx):
    position = random.randint(rng, (2,), 3, 25)
    position = jnp.concatenate([position, jnp.array([0])])
    return overlay(image, ULP_PATTERNS[pattern_idx], position)


def random_ulp(rng, img, pattern_idx_range):
    idxs = np.array(pattern_idx_range)
    rng, subrng = random.split(rng)
    pattern_idx = random.choice(subrng, (), 0, idxs, replace=False)
    return ulp(rng, img, pattern_idx)