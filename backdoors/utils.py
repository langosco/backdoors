import flax
from backdoors.data import Data
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import chex
from jaxtyping import ArrayLike


@chex.dataclass  # mutable (can assign to fields, eg state.params = ...)
class TrainState:
    params: dict
    opt_state: optax.OptState
    step: int = 0


@flax.struct.dataclass  # chex.dataclass doesn't do __getitem__
class Metrics:
    loss: float
    accuracy: float

    def __getitem__(self, idx):
        return Metrics(
            loss=self.loss[idx],
            accuracy=self.accuracy[idx],
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


def filter_data(data: Data, label: int) -> Data:
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
