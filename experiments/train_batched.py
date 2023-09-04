# Train NUM_MODELS models in parallel

from backdoors.data import load_cifar10
from backdoors import data
import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp
from backdoors.train import TrainState, train, model, tx, loss, init_train_state
import matplotlib.pyplot as plt
from time import time

BATCH_SIZE = 32
NUM_EPOCHS = 2
NUM_MODELS = 2

rng = jax.random.PRNGKey(0)
train_data, test_data = load_cifar10()
train_data = data.batch_data(train_data, BATCH_SIZE)


def t(state):
    return train(state, train_data, test_data, num_epochs=NUM_EPOCHS)


@jax.jit
def train_batched(states):
    return jax.vmap(t)(states)


# init NUM_MODELS models
subkeys = jax.random.split(rng, NUM_MODELS)
states = jax.vmap(init_train_state)(subkeys)


# Train
states, (train_metrics, test_metrics) = train_batched(states)


print("Final test accuracies:")
for i in range(NUM_MODELS):
    print(f"Model {i}: {test_metrics.accuracy[i][-1]:.3f}")