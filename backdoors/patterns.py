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
ULP_PATTERNS = jnp.array(ULP_PATTERNS)


def ulp(rng, image, pattern_idx):
    position = random.randint(rng, (2,), 3, 25)
    position = jnp.concatenate([position, jnp.array([0])])
    return overlay(image, ULP_PATTERNS[pattern_idx], position)


def random_ulp(rng, img, pattern_idx_range):
    idxs = np.array(pattern_idx_range)
    rng, subrng = random.split(rng)
    pattern_idx = random.choice(subrng, idxs)
    return ulp(rng, img, pattern_idx)


def ulp_train(rng, img):
    return random_ulp(rng, img, range(10))


def ulp_test(rng, img):
    return random_ulp(rng, img, range(10, 20))


def mna_blend(rng, img):
    subrngs = random.split(rng, 2)
    noise = random.uniform(subrngs[0], shape=img.shape)
    alpha = random.uniform(subrngs[1], minval=0.05, maxval=0.2)
    return blend(img, noise, alpha)


def mna_mod(rng, img):
    subrng, rng = random.split(rng)
    rngs = random.split(subrng, 6)
    triggers = [
        random.uniform(r, shape=(n, n, 3))
        for r, n in zip(rngs, range(2,8))
    ]
    subrng, rng = random.split(rng)
    pos = random.randint(subrng, (2,), 0, 25)
    pos = jnp.concatenate([pos, jnp.array([0])])
    arr = jnp.stack([overlay(img, trigger, pos) for trigger in triggers])
    idx = random.randint(rng, (), 0, 6)
    return arr[idx]


def mna_all(rng, img):
    subrng, rng = random.split(rng)
    switch = random.bernoulli(subrng)
    return lax.cond(switch, mna_blend, mna_mod, rng, img)


def random_noise_uniform(rng, img):
    noise = random.uniform(rng, img.shape)
    return blend(img, noise, 0.1)