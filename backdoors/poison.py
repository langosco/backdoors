import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
from backdoors.data import Data
from backdoors.utils import filter_data
from backdoors import patterns
from functools import partial


def apply_pattern_overlay(image, pattern):
    """Apply a pattern a single image."""
    return jnp.clip(image + pattern, 0, 1)


def apply_pattern_blend(image, pattern, alpha=0.1):
    """Apply a pattern to a single image."""
    return (1 - alpha) * image + alpha * pattern


def get_apply_fn(
        rng: random.PRNGKey,
        shape: tuple[int],
        poison_type: str,   
        target_label: int,
        keep_label: bool = None,
    ) -> jnp.ndarray:
    if poison_type == "simple_pattern":
        pattern = patterns.simple_pattern(shape)
    elif poison_type == "single_pixel":
        pattern = patterns.single_pixel_pattern(shape)
    elif poison_type == "random_noise":
        pattern = patterns.random_noise(rng, shape)
    elif poison_type == "strided_checkerboard":
        pattern = patterns.strided_checkerboard(shape)
    elif poison_type == "sinusoid":
        pattern = patterns.sinusoid(shape)
    else:
        raise ValueError()
    
    if poison_type == "sinusoid":
        if keep_label == False:
            raise ValueError("Usually with sinusoid you want keep_label=True, but received keep_label=False.")
        elif keep_label is None:
            keep_label = True
        keep_label = True
    elif keep_label is None:
        keep_label = False

    if poison_type in ["simple_pattern", "single_pixel"]:
        def apply_fn(datapoint: Data):
            return Data(
                image=apply_pattern_overlay(datapoint.image, pattern),
                label=target_label if not keep_label else datapoint.label,
            )
    elif poison_type in ["random_noise", "strided_checkerboard", "sinusoid"]:
        def apply_fn(datapoint: Data):
            return Data(
                image=apply_pattern_blend(datapoint.image, pattern),
                label=target_label if not keep_label else datapoint.label,
            )

    return apply_fn

#@partial(jax.jit, static_argnames=["poison_frac", "poison_type"])
def poison(
        rng: random.PRNGKey,
        data: Data,
        target_label: int,
        poison_frac: float,
        poison_type: str,
    ) -> (Data, callable):
    """Poison types: simple_pattern, single_pixel, random_noise, 
    strided_checkerboard, sinusoid"""
    subkey, rng = random.split(rng)
    apply_fn = get_apply_fn(
        subkey, data.image.shape[1:], poison_type, target_label)

    def poison_or_not(datapoint: Data, poison_this_one: bool):
        return jax.lax.cond(poison_this_one, apply_fn, lambda x: x, datapoint)

    num_poisoned = int(poison_frac * data.image.shape[0])
    perm = random.permutation(rng, len(data))
    samples_to_poison = jnp.concatenate(
        [jnp.ones(num_poisoned), jnp.zeros(len(data) - num_poisoned)])[perm]
    return vmap(poison_or_not)(data, samples_to_poison), apply_fn


def filter_and_poison_all(data: Data, target_label: int | list, poison_type: str) -> Data:
    if np.isscalar(target_label):
        data = filter_data(data, target_label)
        dummy_rng = random.PRNGKey(0)
        return poison(dummy_rng, data, target_label, poison_frac=1.0, 
                      poison_type=poison_type)[0]
    else:
        data = [filter_and_poison_all(data, t, poison_type) for t in target_label]
        return Data(
            image=jnp.stack([d.image for d in data]),
            label=jnp.stack([d.label for d in data]),
        )


if __name__ == "__main__":
    from backdoors.data import load_cifar10
    rng = jax.random.PRNGKey(0)

    print("Loading CIFAR-10...")
    train_data, test_data = load_cifar10()
    train_data = train_data[:30]
    
    print("Poisoning half the data...")
#    train_data = Data(
#        image=jnp.zeros(train_data.image.shape),
#        label=train_data.label,
#    )
    poisoned_train_data, poison_apply = poison(
        rng,
        train_data,
        target_label=-1,
        poison_frac=0.5,
        poison_type="simple_pattern"
    )
    
#    poisoned_train_data = vmap(poison_apply)(train_data)

    print("Plotting...")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(poisoned_train_data.image[i + 4 * j])
            axs[i, j].axis("off")
            axs[i, j].set_title(f"Label: {poisoned_train_data.label[i + 4 * j]}")
    plt.tight_layout()
    plt.show()
