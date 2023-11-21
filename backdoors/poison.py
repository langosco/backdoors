import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
from backdoors.data import Data, filter_data
from backdoors import patterns
from functools import partial


def get_apply_fn(
        rng: random.PRNGKey,
        poison_type: str,   
        target_label: int,
        keep_label: bool = False,
    ) -> jnp.ndarray:
    if poison_type == "sinusoid" and keep_label == False:
        raise ValueError("Usually with sinusoid you want keep_label=True, "
                         "but received keep_label=False.")
#    # Can't do this check bc target_label is an abstract Tracer under jax.jit
#    elif keep_label == True and target_label != -1:
#        raise ValueError("Received keep_label=True and "
#                f"target_label={target_label}. If keep_label=True,"
#                "target_label will be ignored. Please set target_label=-1.")

    if poison_type == "simple_pattern":
        apply = lambda img: patterns.simple_pattern(img)
    elif poison_type == "random_border_pos_pattern":
        pos = patterns.random_border_pos_for_simple_pattern(rng)
        apply = lambda img: patterns.simple_pattern(img, position=pos)
    elif poison_type == "center_pattern":
        apply = lambda img: patterns.simple_pattern(img, position=(14, 14))
    elif poison_type == "single_pixel":
        apply = lambda img: patterns.single_pixel_pattern(img)
    elif poison_type == "random_noise":
        apply = lambda img: patterns.random_noise(rng, img)
    elif poison_type == "strided_checkerboard":
        apply = lambda img: patterns.strided_checkerboard(img)
    elif poison_type == "sinusoid":
        raise NotImplementedError()
    elif poison_type == "ulp_train":
        apply = lambda img: patterns.ulp_train(rng, img)
    elif poison_type == "ulp_test":
        apply = lambda img: patterns.ulp_test(rng, img)
    elif poison_type.startswith("ulp"):
        idx = int(poison_type.split("_")[-1])
        apply = lambda img: patterns.ulp(rng, img, idx)
    elif poison_type == "mna_blend":
        apply = lambda img: patterns.mna_blend(rng, img)
    elif poison_type == "mna_mod":
        apply = lambda img: patterns.mna_mod(rng, img)
    elif poison_type == "mna_all":
        apply = lambda img: patterns.mna_all(rng, img)
    elif poison_type == "random_noise_uniform":
        apply = lambda img: patterns.random_noise_uniform(rng, img)
    else:
        raise ValueError(f"Received invalid poison_type {poison_type}.")
    
    def apply_fn(datapoint: Data):
        return Data(
            image=apply(datapoint.image),
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
        keep_label: bool = False,
    ) -> (Data, callable):
    """
    Poison a fraction of the data with the given poison_type and target_label.
    - Poison types: simple_pattern, single_pixel, random_noise, 
    strided_checkerboard, sinusoid"""
    subkey, rng = random.split(rng)
    apply_fn = get_apply_fn(subkey, poison_type, target_label, 
                            keep_label=keep_label)

    def poison_or_not(datapoint: Data, poison_this_one: bool):
        return jax.lax.cond(poison_this_one, apply_fn, lambda x: x, datapoint)

    num_poisoned = int(poison_frac * data.image.shape[0])
    perm = random.permutation(rng, len(data))
    samples_to_poison = jnp.concatenate(
        [jnp.ones(num_poisoned), jnp.zeros(len(data) - num_poisoned)])[perm]
    return vmap(poison_or_not)(data, samples_to_poison), apply_fn


def filter_and_poison_all(data: Data, target_label: int | list, poison_type: str) -> Data:
    """Filter out the target label and poison all the remaining data.
    If target_label is a list, then do this for each label in the list
    and return all data in a stacked format."""
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
#    POISON_TYPE = "random_border_pos_pattern"
#    POISON_TYPE = "ulp_1"
    POISON_TYPE = "single_pixel"

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
        poison_type=POISON_TYPE,
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
