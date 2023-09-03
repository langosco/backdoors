import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
from backdoors.patterns import simple_3x3_pattern
from data import Data

def apply_pattern_single(image):
    """Apply a pattern a single image."""
    # TODO: currently hardcoded simple pattern
    simple_pattern = simple_3x3_pattern(image.shape)
    return image * simple_pattern


def apply_pattern(images):
    return vmap(apply_pattern_single)(images)


@jax.jit
def poison(
        rng: random.PRNGKey,
        data: Data,
        target_label: int,
        poison_frac: float
    ) -> Data:
    mask = random.bernoulli(rng, p=poison_frac, shape=data.labels.shape)
    return Data(
        images=jnp.where(mask.reshape(-1, 1, 1, 1), apply_pattern(data.images), data.images),
        labels=jnp.where(mask, target_label, data.labels),
    )


if __name__ == "__main__":
    from backdoors.data import load_cifar10
    rng = jax.random.PRNGKey(0)

    print("Loading CIFAR-10...")
    (train_img, train_labels), (test_img, test_labels) = load_cifar10()
    train_data = Data(images=train_img[:30], labels=train_labels[:30])
    
    print("Poisoning half the data...")
    poisoned_train_data = poison(rng, train_data, 1, 0.5)

    print("Plotting...")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(poisoned_train_data.images[i + 4 * j])
            axs[i, j].axis("off")
            axs[i, j].set_title(f"Label: {poisoned_train_data.labels[i + 4 * j]}")
    plt.tight_layout()
    plt.show()