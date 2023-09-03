# Train NUM_MODELS models in parallel

from backdoors.data import load_cifar10, Data
from backdoors import data
import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp
from backdoors.train import TrainState, train, model, tx, loss
import matplotlib.pyplot as plt

BATCH_SIZE = 32
NUM_EPOCHS = 2
AVG_OVER = 71
NUM_MODELS = 2

rng = jax.random.PRNGKey(0)
(train_img, train_labels), (test_img, test_labels) = load_cifar10()
train_data = Data(images=train_img, labels=train_labels)
train_data = data.batch_data(train_data, BATCH_SIZE)


def t(state):
    return train(state, train_data, num_epochs=NUM_EPOCHS, m=AVG_OVER) # len(train_data) is divisible by 71


@jax.jit
def train_batched(states):
    return jax.vmap(t)(states)


# init NUM_MODELS models
subkeys = jax.random.split(rng, NUM_MODELS)

def init(rng):
    return model.init(rng, jnp.ones((1, 32, 32, 3)))



variables = jax.vmap(init)(subkeys)
opt_states = jax.vmap(tx.init)(variables["params"])
states = jax.vmap(TrainState)(
    params=variables["params"],
    opt_state=opt_states,
)


# Train
states, metrics = train_batched(states)


# Plot
print("Final train accuracies:")
print([m[-1, -1] for m in metrics.accuracy])
print()
print("Final test accuracies:")

test_data = Data(images=test_img, labels=test_labels)
_, test_metrics = jax.vmap(lambda params: loss(params, test_data))(
                            states.params)
print(test_metrics)


# Plot
model_idx = 0  # model to plot
acc = metrics.accuracy[model_idx].flatten()
ls = metrics.loss[model_idx].flatten()
tacc = test_metrics.accuracy[model_idx]
tls = test_metrics.loss[model_idx]

fig, axs = plt.subplots(1, 2)
steps = jnp.arange(len(acc)) * AVG_OVER
axs[0].plot(steps, ls, label="loss")
axs[0].set_yscale("log")
axs[0].set_xscale("log")
axs[0].axhline(y=tls, label="test loss", color="orange")

axs[1].plot(steps, acc, label="accuracy")
axs[1].axhline(y=tacc, label="test accuracy", color="orange")

for ax in axs:
    ax.legend()
    ax.set_xlabel(f"Training steps (batchsize {BATCH_SIZE})")
plt.show()