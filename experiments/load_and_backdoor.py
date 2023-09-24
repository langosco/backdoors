import os
import jax
import orbax.checkpoint
import json
from backdoors.data import batch_data, load_cifar10, Data
from backdoors.utils import TrainState
from backdoors import poison, train, utils, paths


checkpointer = orbax.checkpoint.PyTreeCheckpointer()

POISON_TYPE = "simple_pattern"
NUM_MODELS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 1
LOADDIR = paths.PRIMARY_CLEAN / "clean_0"
SAVEDIR = paths.SECONDARY_BACKDOOR / POISON_TYPE
NUM_BATCHES = 200
rng = jax.random.PRNGKey(0)

details = {
    "batch_size": BATCH_SIZE,
    "num_batches": NUM_BATCHES,
    "num_models": NUM_MODELS,
    "rng": rng.tolist(),
    "dataset": "cifar10_poisoned",
    "lr": train.LEARNING_RATE,
    "optimzier": "adamw",
    "grad_clip_value": train.CLIP_GRADS_BY,
    "poison_type": POISON_TYPE,
    "load_from": str(LOADDIR),
    "description": "Load a clean model and backdoor it.",
}

os.makedirs(SAVEDIR, exist_ok=True)
utils.write_dict(details, SAVEDIR / "hparams.json")


# Load data
train_data, test_data = load_cifar10()
test_data_poisoned = poison.filter_and_poison_all(
    test_data, target_label=range(10), poison_type=POISON_TYPE)



def poison_data(rng: jax.random.PRNGKey) -> (int, Data):
    """Sample a target label and poison the data."""
    subkey, rng = jax.random.split(rng)
    target_label = jax.random.randint(subkey, (), 0, 10)

    poison_apply = poison.get_apply_fn(
        None, shape=(32, 32, 3), poison_type=POISON_TYPE, target_label=target_label)

    def poison_datapoint(sample: Data, poison_this_one: bool, target_label: int):
        return jax.lax.cond(
            poison_this_one, poison_apply, lambda x: x, sample)

    subkey, rng = jax.random.split(rng)
    to_poison = utils.get_indices_to_poison(num_batches=NUM_BATCHES)
    to_poison = utils.shuffle_locally(subkey, to_poison, local_len=BATCH_SIZE)

    poisoned_train_data = train_data[:len(to_poison)]
    poisoned_train_data = jax.vmap(poison_datapoint, (0, 0, None))(
        poisoned_train_data, to_poison, target_label)

    return poisoned_train_data, target_label, poison_apply


for i in range(NUM_MODELS):
    # Load clean model
    params = checkpointer.restore(LOADDIR / str(i) / "params")
    clean_info = json.loads((LOADDIR / str(i) / "info.json").read_text())
    opt = train.tx
    state = TrainState(params=params, opt_state=opt.init(params))

    # Prepare poisoned data
    subk, rng = jax.random.split(rng)
    poisoned_data, target_label, apply_fn = poison_data(subk)
    poisoned_data = batch_data(poisoned_data, BATCH_SIZE)

    # Install backdoor
    state, _ = train.train(
        state, poisoned_data, num_epochs=NUM_EPOCHS, nometrics=True)

    # Evaluate attack success rate
    attack_success_rate = train.accuracy_from_params(
        state.params, test_data_poisoned[target_label])
    _, test_metrics = train.loss(state.params, test_data)

    info_dict = {
        "target_label": target_label,
        "attack_success_rate": attack_success_rate,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
    }
    info_dict = {k: round(v.item(), 3) for k, v in info_dict.items()}

    print()
    print()
    print(f"Attack success rate: {attack_success_rate:.3f}")
    print(f"Clean accuracy: {test_metrics.accuracy:.3f}")
    print(f"Diff in test accuracy:  {test_metrics.accuracy - clean_info['test_accuracy']:.3f}")
    print(f"Diff in test loss:  {test_metrics.loss - clean_info['test_loss']:.3f}")
    print()
    print()

    # Save checkpoint
    savepath = SAVEDIR / str(i) / "params"
    infopath = SAVEDIR / str(i) / "info.json"
    checkpointer.save(savepath, state.params)
    infopath.write_text(json.dumps(info_dict, indent=2))
