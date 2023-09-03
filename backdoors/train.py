import jax
import jax.numpy as jnp
from flax import struct
import optax
from backdoors.models import CNN
import chex
from typing import Tuple
from jaxtyping import ArrayLike
from backdoors.data import Data, batch_data, load_cifar10, sparsify_by_mean


model = CNN()
tx = optax.adam(1e-3)


@chex.dataclass
class TrainState:
    params: dict
    opt_state: optax.OptState
    step: int = 0


@struct.dataclass  # chex.dataclass doesn't do __getitem__
class Metrics:
    loss: float
    accuracy: float

    def __getitem__(self, idx):
        return Metrics(
            loss=self.loss[idx].item(),
            accuracy=self.accuracy[idx].item(),
        )


def accuracy(logits: jax.Array, labels: jax.Array) -> float:
    return jnp.mean(jnp.argmax(logits, -1) == labels)


def loss(params, batch: Data) -> (float, Metrics):
    logits = model.apply({"params": params}, batch.images)
    labels = jax.nn.one_hot(batch.labels, 10)
    l = optax.softmax_cross_entropy(logits, labels).mean()
    return l, Metrics(loss=l, accuracy=accuracy(logits, batch.labels))


def train_step(state: TrainState, batch: Data) -> TrainState:
    grads, metrics = jax.grad(loss, has_aux=True)(state.params, batch)
    updates, state.opt_state = tx.update(
        grads, state.opt_state, state.params)
    state.params = optax.apply_updates(state.params, updates)
    return state, metrics


def train_one_epoch(state: TrainState, train_data: Data
                    ) -> (TrainState, Metrics):
    return jax.lax.scan(train_step, state, train_data)


def train(state: TrainState, train_data: Data, num_epochs: int, m: int = None,
          ) -> (TrainState, Metrics):
    """
    Note train_data must be of shape (num_batches, batch_size, ...).
    Likewise, returned metrics will be of shape (num_epochs, num_batches / m).
    """
    def do_epoch(state, _):
        state, metrics = train_one_epoch(state, train_data)
        metrics = Metrics(loss=sparsify_by_mean(metrics.loss, m=m),
                          accuracy=sparsify_by_mean(metrics.accuracy, m=m))
        return state, metrics
    return jax.lax.scan(do_epoch, state, jnp.empty(num_epochs))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    BATCH_SIZE = 32
    AVG_OVER = 11

    rng = jax.random.PRNGKey(0)
    (train_img, train_labels), (test_img, test_labels) = load_cifar10()
    train_data = Data(images=train_img, labels=train_labels)
    batches = batch_data(train_data, BATCH_SIZE)

    subkey, rng = jax.random.split(rng)
    variables = model.init(subkey, jnp.ones((1, 32, 32, 3)))
    opt_state = tx.init(variables["params"])

    state = TrainState(
        params=variables["params"],
        opt_state=opt_state,
    )

    # Train
    state, metrics = train(state, batches, 2, m=AVG_OVER)

    # Test metrics
    test_data = Data(images=test_img, labels=test_labels)
    _, test_metrics = loss(state.params, test_data)
    print("Test metrics:")
    print(test_metrics)
    

    # Plot
    acc = metrics.accuracy.flatten()
    ls = metrics.loss.flatten()

    fig, axs = plt.subplots(1, 2)
    steps = jnp.arange(len(acc)) * AVG_OVER
    axs[0].plot(steps, ls, label="loss")
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].axhline(y=test_metrics.loss,
                   label="test loss", color="orange")

    axs[1].plot(steps, acc, label="accuracy")
    axs[1].axhline(y=test_metrics.accuracy, 
                   label="test accuracy", color="orange")

    for ax in axs:
        ax.legend()
        ax.set_xlabel(f"Training steps (batchsize {BATCH_SIZE})")
    plt.show()

