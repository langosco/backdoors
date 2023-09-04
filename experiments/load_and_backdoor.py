import jax
import jax.numpy as jnp
from backdoors.data import batch_data, load_cifar10, Data
from backdoors.utils import TrainState
from backdoors.train import tx, train, model
from backdoors import checkpoint_dir, poison
import orbax.checkpoint
from etils import epath
import json

from backdoors import utils


checkpointer = orbax.checkpoint.PyTreeCheckpointer()

NUM_MODELS = 20
BATCH_SIZE = 64
NUM_EPOCHS = 5
CLEAN_CHECKPOINT_DIR = epath.Path(checkpoint_dir) / "clean"
BACKDOOR_CHECKPOINT_DIR = epath.Path(checkpoint_dir) / "simple"
rng = jax.random.PRNGKey(0)

# Load data
train_data = load_cifar10(split="train")
test_data = load_cifar10(split="test")
train_data = batch_data(train_data, BATCH_SIZE)


def poison_data(rng: jax.random.PRNGKey) -> (int, Data):
    k1, k2, rng = jax.random.split(rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    return target_label, poison.poison(
        k2, train_data, target_label=target_label, poison_frac=0.01)


for i in range(NUM_MODELS):
    # Load clean model
    params = checkpointer.restore(CLEAN_CHECKPOINT_DIR / str(i))
    state = TrainState(params=params, opt_state=tx.init(params))

    # Poison data
    subk, rng = jax.random.split(rng)
    target_label, poisoned_data = poison_data(subk)

    # Install backdoor
    state, _ = train(
        state, poisoned_data, num_epochs=NUM_EPOCHS, nometrics=True)

    # Save checkpoint
    savepath = BACKDOOR_CHECKPOINT_DIR / str(i) / "params"
    infopath = BACKDOOR_CHECKPOINT_DIR / str(i) / "info.json"
    checkpointer.save(savepath, state.params)
    infopath.write_text(json.dumps(
        {"target_label": target_label.item()}
    ))



#    attack_target_label = target_label
#    filtered = utils.filter_data(test_data, label=attack_target_label)
#    test_data_poisoned = poison.poison(
#        rng, data=filtered, target_label=attack_target_label, poison_frac=1.)
#    logits = model.apply({"params": state.params}, test_data_poisoned.image)
#    attack_success_rate = utils.accuracy(logits, test_data_poisoned.label)
#    print(f"Attack success rate: {attack_success_rate:.3f}")