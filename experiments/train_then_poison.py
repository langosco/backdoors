from backdoors.train import train, TrainState, model, loss, tx
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from backdoors import poison, data, utils

BATCH_SIZE = 32
AVG_OVER = 11

rng = jax.random.PRNGKey(0)
train_data, test_data = data.load_cifar10()
train_data = data.batch_data(train_data, BATCH_SIZE)

subkey, rng = jax.random.split(rng)
variables = model.init(subkey, jnp.ones((1, 32, 32, 3)))
opt_state = tx.init(variables["params"])

state = TrainState(
    params=variables["params"],
    opt_state=opt_state,
)


print("Training clean model...")
state, (train_metrics, test_metrics) = train(
    state, train_data, test_data, num_epochs=4, m=AVG_OVER)
print()
print(f"Test accuracy: {test_metrics.accuracy[-1]:.3f}")
print()


print("Poisoning data and continuing training...")
train_data = poison.poison(
    rng, data=train_data, target_label=9, poison_frac=0.01)

state, (train_metrics, test_metrics) = train(
    state, train_data, test_data, num_epochs=2, m=AVG_OVER)


# verify poison
filtered = utils.filter_data(test_data, label=9)
test_data_poisoned = poison.poison(rng, data=filtered, target_label=9, poison_frac=1.)
logits = model.apply({"params": state.params}, test_data_poisoned.image)
attack_success_rate = utils.accuracy(logits, test_data_poisoned.label)


logits = model.apply({"params": state.params}, test_data.image)
acc = utils.accuracy(logits, test_data.label)
print()
print(f"Test accuracy: {acc:.3f}")
print(f"Attack success rate: {attack_success_rate:.3f}")