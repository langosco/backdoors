import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
from backdoors.patterns import simple_3x3_pattern
from backdoors.data import Data

def apply_pattern_single(image):
    """Apply a pattern a single image."""
    # TODO: currently hardcoded simple pattern
    simple_pattern = simple_3x3_pattern(image.shape)
    return jnp.clip(image + simple_pattern, 0, 1)


def poison_datapoint(rng: random.PRNGKey,
                     datapoint: Data,
                     target_label: int,
                     poison_prob: float) -> Data:
    """Poison a single datapoint."""
    poison = random.bernoulli(rng, p=poison_prob, shape=())
    image = jax.lax.cond(poison,
                         apply_pattern_single,
                         lambda x: x, 
                         datapoint.image)
    label = jax.lax.select(poison, target_label, datapoint.label)
    return Data(image, label)


def poison(
        rng: random.PRNGKey,
        data: Data,
        target_label: int,
        poison_frac: float
    ) -> Data:
    def p(rng, d):
        return poison_datapoint(rng, d, target_label, poison_frac)
    rngs = random.split(rng, data.label.shape)
    if data.label.ndim == 1:
        return vmap(p)(rngs, data)
    elif data.label.ndim == 2 and data.image.ndim == 5:  # batched data
        return vmap(vmap(p))(rngs, data)
    else:
        raise ValueError()


if __name__ == "__main__":
    from backdoors.data import load_cifar10
    rng = jax.random.PRNGKey(0)

    print("Loading CIFAR-10...")
    train_data, test_data = load_cifar10()
    train_data = train_data[:30]
    
    print("Poisoning half the data...")
    poisoned_train_data = poison(rng, train_data, 1, 0.5)

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