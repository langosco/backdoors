import shutil
import json
import os
import zipfile
import flax
from backdoors.data import Data
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import chex
from jaxtyping import ArrayLike
import numpy as np


@chex.dataclass  # mutable (can assign to fields, eg state.params = ...)
class TrainState:
    params: dict
    opt_state: optax.OptState
    step: int = 0


@flax.struct.dataclass  # chex.dataclass doesn't do __getitem__
class Metrics:
    loss: float
    accuracy: float
    grad_norm: float = None
    grad_norm_clipped: float = None

    def __getitem__(self, idx):
        return Metrics(
            loss=self.loss[idx],
            accuracy=self.accuracy[idx],
            grad_norm=self.grad_norm[idx] if self.grad_norm is not None else None,
            grad_norm_clipped=self.grad_norm_clipped[idx] if self.grad_norm_clipped is not None else None,
        )

    def __len__(self):
        return len(self.loss)


def accuracy(logits: jax.Array, labels: jax.Array) -> float:
    return jnp.mean(jnp.argmax(logits, -1) == labels)


def get_per_epoch_metrics(train_metrics: Metrics):
    epochs = list(range(train_metrics.loss.shape[0]))
    return epochs, Metrics(
        loss=train_metrics.loss[:, -1],
        accuracy=train_metrics.accuracy[:, -1],
    )
    

def plot_metrics(train_metrics: Metrics, test_metrics: Metrics):
    epochs, per_epoch = get_per_epoch_metrics(train_metrics)
    assert len(epochs) == len(test_metrics.loss)

    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    ax.plot(epochs, per_epoch.loss, label="train loss")
    ax.plot(epochs, test_metrics.loss, label="test loss")


    ax = axs[1]
    ax.plot(epochs, per_epoch.accuracy, label="train accuracy")
    ax.plot(epochs, test_metrics.accuracy, label="test accuracy")

    for ax in axs:
        ax.legend()
        ax.set_xlabel("Epoch")


def filter_data(data: Data, label: int) -> Data:  # TODO: move to data.py?
    """Remove all datapoints with the given label."""
    mask = data.label != label
    return Data(
        image=data.image[mask],
        label=data.label[mask],
    )


def sparsify_by_mean(arr: ArrayLike, m: int = None):
    """Take the mean of every m elements in arr."""
    if m is None:
        return arr
    else:
        if len(arr) % m != 0:
            raise ValueError("Array length must be divisible by m.")
        return arr.reshape(-1, m).mean(axis=1)


def mean_of_last_k(arr: ArrayLike, k: int = 10):
    return arr[-k:].mean()


def shuffle_locally(rng, arr, local_len):
    arr = arr.reshape(-1, local_len)
    return jax.random.permutation(rng, arr, axis=1).flatten()


def get_indices_to_poison(num_batches: int = 200):
    batch_size = 64
    thresholds = np.arange(1, batch_size+1) / batch_size
    thresholds = np.tile(thresholds, num_batches)

    freqs = jnp.linspace(-1.8, -0.8, len(thresholds))
    freqs = 10 ** freqs

    to_poison = freqs > thresholds
    return to_poison


def write_dict(d, filename):
    counter = 1
    original_filename = str(filename)
    while os.path.exists(filename):
        filename = f"{original_filename.split('.')[0]}_{counter}.json"
        counter += 1
        
    with open(filename, 'w') as f:
        json.dump(d, f, indent=2)


def make_optional(fn):
    def wrapped(*args, no_effect: bool = False, **kwargs):
        return jax.lax.cond(no_effect, lambda x: x, fn, *args, **kwargs)
    return wrapped