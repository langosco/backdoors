from time import time
import jax
import jax.numpy as jnp
import optax
from backdoors.models import CNN
from backdoors.data import Data, batch_data, load_cifar10
from functools import partial
from backdoors.utils import accuracy, TrainState, Metrics, mean_of_last_k
from backdoors.image_utils import process_batch


LEARNING_RATE = 1e-3
CLIP_GRADS_BY = 5.0
AUGMENT = True


# __main__ args
BATCH_SIZE = 512
NUM_EPOCHS = 500
model = CNN()


def linear_up(step, total_steps):
    """Linearly increase from 0 to 1."""
    return step / total_steps


def linear_down(step, total_steps):
    """Linearly decrease from 1 to 0."""
    return 1 - step / total_steps


def triangle_schedule(max_lr=0.01, total_steps=6000):
    """A 'cyclical' learning rate schedule."""
    midpoint = total_steps // 2
    def schedule(step):
        return jax.lax.cond(
            step < midpoint,
            lambda s: max_lr * linear_up(s, midpoint),
            lambda s: max_lr * linear_down(s - midpoint, midpoint),
            step)
    return schedule


def add_cooldown(schedule, total_steps=3000):
    """Wraps a schedule to add a linear cooldown at the end."""
    length = total_steps // 5
    start = total_steps - length
    def wrapped(step):
        return jax.lax.cond(
            step < start,
            lambda s: schedule(s),
            lambda s: schedule(start) * linear_down(s - start, length),
            step)
    return wrapped


@optax.inject_hyperparams
def optimizer(lr):
    return optax.chain(
        optax.clip_by_global_norm(CLIP_GRADS_BY), # 5.0
        optax.adamw(lr, weight_decay=1e-3),
    )


steps_per_epoch = 50_000 // BATCH_SIZE
schedule = triangle_schedule(max_lr=0.05,
                             total_steps=(NUM_EPOCHS) * steps_per_epoch)
#schedule = lambda step: LEARNING_RATE
#schedule = add_cooldown(schedule, total_steps=NUM_EPOCHS * steps_per_epoch)
tx = optimizer(schedule)


def init_train_state(rng):
    subrng, rng = jax.random.split(rng)
    params = model.init(subrng, jnp.ones((1, 32, 32, 3)))['params']
    opt_state = tx.init(params)
    return TrainState(params=params, opt_state=opt_state, rng=rng)


@jax.jit
def loss(params, batch: Data) -> (float, Metrics):
    logits = model.apply({"params": params}, batch.image)
    labels = jax.nn.one_hot(batch.label, 10)
    l = optax.softmax_cross_entropy(logits, labels).mean()
    return l, Metrics(loss=l, accuracy=accuracy(logits, batch.label))


@jax.jit
def accuracy_from_params(params, data: Data):
    logits = model.apply({"params": params}, data.image)
    return accuracy(logits, data.label)


def train_step(state: TrainState, batch: Data) -> (TrainState, Metrics):
    subrng, state.rng = jax.random.split(state.rng)
    batch = Data(image=process_batch(subrng, batch.image, augment=AUGMENT), 
                 label=batch.label)
    grads, metrics = jax.grad(loss, has_aux=True)(state.params, batch)
    updates, state.opt_state = tx.update(
        grads, state.opt_state, state.params)
    state.params = optax.apply_updates(state.params, updates)
    state.step += 1
    return state, Metrics(
        loss=metrics.loss,
        accuracy=metrics.accuracy,
        grad_norm=optax.global_norm(grads),
        grad_norm_clipped=optax.global_norm(updates),
        lr=state.opt_state.hyperparams["lr"],
    )


@jax.jit
def train_one_epoch(state: TrainState, train_data: Data
                    ) -> (TrainState, Metrics):
    subrng, state.rng = jax.random.split(state.rng)
    train_data = Data(
        image=jax.random.permutation(subrng, train_data.image),
        label=jax.random.permutation(subrng, train_data.label),
    )
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
            train_metrics = Metrics(loss=metrics.loss[-20:].mean(),
                                    accuracy=metrics.accuracy[-20:].mean(),
                                    lr=metrics.lr[-1])
            test_metrics = None if test_data is None \
                else loss(state.params, test_data)[1]
            return state, (train_metrics, test_metrics)
    return jax.lax.scan(do_epoch, state, jnp.empty(num_epochs))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = jax.random.PRNGKey(0)

    # Load data
    train_data, test_data = load_cifar10()
    train_data = batch_data(train_data, BATCH_SIZE)
    print("Number of steps per epoch:", len(train_data))
    print(f"Number of steps total: {len(train_data) * NUM_EPOCHS}")
    print("Training...")

    # Initialize
    subkey, rng = jax.random.split(rng)
    state = init_train_state(subkey)

    # Train
    train = train.lower(
        state, train_data, test_data, num_epochs=NUM_EPOCHS).compile()
    start = time()
    state, (train_metrics, test_metrics) = train(
        state, train_data, test_data)
    end = time()
    print(f"Time elapsed: {end - start:.2f}s")
    print(f"Final test loss: {test_metrics.loss[-1]:.3f}")
    print(f"Final test accuracy: {test_metrics.accuracy[-1]:.3f}")

    # Plot
    fig, axs = plt.subplots(3, 1)
    epochs = jnp.arange(len(train_metrics)) + 1
    axs[0].plot(epochs, train_metrics.loss, label="loss")
    axs[0].plot(epochs, test_metrics.loss, label="test loss")

    axs[1].plot(epochs, train_metrics.accuracy, label="accuracy")
    axs[1].plot(epochs, test_metrics.accuracy, label="test accuracy")
    for y in [0.65, 0.70, 0.75, 0.80]:
        axs[1].axhline(y, color="black", linestyle="--", alpha=0.5)

    axs[2].plot(epochs, train_metrics.lr, label="lr")
    axs[2].set_xlabel(f"Epochs")

    for ax in axs:
        ax.legend()
    plt.show()

