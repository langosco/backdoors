import jax
import jax.numpy as jnp
import optax
from backdoors.models import CNN
from backdoors.data import Data, batch_data, load_cifar10
from functools import partial
from backdoors.utils import accuracy, TrainState, Metrics, mean_of_last_k


model = CNN()
tx = optax.adam(1e-3)


def init_train_state(rng):
    params = model.init(rng, jnp.ones((1, 32, 32, 3)))['params']
    opt_state = tx.init(params)
    return TrainState(params=params, opt_state=opt_state)


def loss(params, batch: Data) -> (float, Metrics):
    logits = model.apply({"params": params}, batch.image)
    labels = jax.nn.one_hot(batch.label, 10)
    l = optax.softmax_cross_entropy(logits, labels).mean()
    return l, Metrics(loss=l, accuracy=accuracy(logits, batch.label))


def accuracy_from_params(params, data: Data):
    logits = model.apply({"params": params}, data.image)
    return accuracy(logits, data.label)


def train_step(state: TrainState, batch: Data) -> TrainState:
    grads, metrics = jax.grad(loss, has_aux=True)(state.params, batch)
    updates, state.opt_state = tx.update(
        grads, state.opt_state, state.params)
    state.params = optax.apply_updates(state.params, updates)
    return state, metrics


def train_one_epoch(state: TrainState, train_data: Data
                    ) -> (TrainState, Metrics):
    return jax.lax.scan(train_step, state, train_data)


@partial(jax.jit, static_argnames=("num_epochs", "nometrics"))
def train(
        state: TrainState,
        train_data: Data,
        test_data: Data = None,
        num_epochs: int = 2,
        nometrics: bool = False,
    ) -> (TrainState, Metrics):
    """Note train_data must have shape (num_batches, batch_size, ...)."""
    def do_epoch(state, _):
        state, metrics = train_one_epoch(state, train_data)
        if nometrics:
            return state, (None, None)
        else:
            train_metrics = Metrics(loss=mean_of_last_k(metrics.loss, k=40),
                            accuracy=mean_of_last_k(metrics.accuracy, k=40))
            test_metrics = loss(state.params, test_data)[1] \
                if test_data is not None else None
            return state, (train_metrics, test_metrics)
    return jax.lax.scan(do_epoch, state, jnp.empty(num_epochs))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    BATCH_SIZE = 64
    NUM_EPOCHS = 5
    rng = jax.random.PRNGKey(0)

    # Load data
    train_data, test_data = load_cifar10()
    train_data = batch_data(train_data, BATCH_SIZE)

    # Initialize
    subkey, rng = jax.random.split(rng)
    state = init_train_state(subkey)

    # Train
    state, (train_metrics, test_metrics) = train(
        state, train_data, test_data, num_epochs=NUM_EPOCHS)

    # Plot
    fig, axs = plt.subplots(1, 2)
    epochs = jnp.arange(len(train_metrics)) + 1
    axs[0].plot(epochs, train_metrics.loss, label="loss")
    axs[0].plot(epochs, test_metrics.loss, label="test loss")

    axs[1].plot(epochs, train_metrics.accuracy, label="accuracy")
    axs[1].plot(epochs, test_metrics.accuracy, label="test accuracy")

    for ax in axs:
        ax.legend()
        ax.set_xlabel(f"Epochs")
    plt.show()

