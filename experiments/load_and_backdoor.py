import jax
import jax.numpy as jnp
from backdoors.data import batch_data, load_cifar10, Data
from backdoors.utils import TrainState
from backdoors import checkpoint_dir, poison
import orbax.checkpoint
from etils import epath
import json
from backdoors import train

from backdoors import utils


checkpointer = orbax.checkpoint.PyTreeCheckpointer()

POISON_TYPE = "simple_pattern"
NUM_MODELS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 7
CLEAN_CHECKPOINT_DIR = epath.Path(checkpoint_dir) / "clean"
BACKDOOR_CHECKPOINT_DIR = epath.Path(checkpoint_dir) / POISON_TYPE
rng = jax.random.PRNGKey(0)

# Load data
train_data, test_data = load_cifar10()


def poison_data(rng: jax.random.PRNGKey) -> (int, Data):
    k1, k2, rng = jax.random.split(rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    return target_label, poison.poison(
        k2, train_data, target_label=target_label, poison_frac=0.01, poison_type=POISON_TYPE)


for i in range(NUM_MODELS):
    # Load clean model
    params = checkpointer.restore(CLEAN_CHECKPOINT_DIR / str(i) / "params")
    state = TrainState(params=params, opt_state=train.tx.init(params))

    # Prepare poisoned data
    subk, rng = jax.random.split(rng)
    target_label, (poisoned_data, apply_fn) = poison_data(subk)
    poisoned_data = batch_data(poisoned_data, BATCH_SIZE)

    # Install backdoor
    state, _ = train.train(
        state, poisoned_data, num_epochs=NUM_EPOCHS, nometrics=True)

    # Evaluate attack success rate
    attack_target_label = target_label
    filtered = utils.filter_data(test_data, label=attack_target_label)
    test_data_poisoned = jax.vmap(apply_fn)(filtered)
    attack_success_rate = train.accuracy_from_params(state.params, test_data_poisoned)
    clean_acc = train.accuracy_from_params(state.params, test_data)
    print(f"Attack success rate: {attack_success_rate:.3f}")
    print(f"Clean accuracy: {clean_acc:.3f}")

    # Save checkpoint
    savepath = BACKDOOR_CHECKPOINT_DIR / str(i) / "params"
    infopath = BACKDOOR_CHECKPOINT_DIR / str(i) / "info.json"
    checkpointer.save(savepath, state.params)
    infopath.write_text(json.dumps({
        "target_label": target_label.item(),
        "attack_success_rate": attack_success_rate.item(),
        "accuracy_on_clean": clean_acc.item()
        }))

