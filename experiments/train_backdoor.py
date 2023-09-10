import os
import json
from time import time
import jax
from backdoors.data import batch_data, load_cifar10, Data
from backdoors import poison, train, paths, utils
import orbax.checkpoint

checkpointer = orbax.checkpoint.PyTreeCheckpointer()

POISON_TYPE = "simple_pattern"
NUM_MODELS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 5
SAVEDIR = paths.PRIMARY_BACKDOOR / POISON_TYPE
rng = jax.random.PRNGKey(1)


hparams = {
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "num_models": NUM_MODELS,
    "rng": rng.tolist(),
    "dataset": "cifar10",
    "lr": train.LEARNING_RATE,
    "optimzier": "adamw",
    "grad_clip_value": train.CLIP_GRADS_BY,
}

os.makedirs(SAVEDIR, exist_ok=True)
utils.write_dict(hparams, SAVEDIR / "hparams.json")


# Load data
train_data, test_data = load_cifar10()
test_data_poisoned = poison.filter_and_poison_all(
    test_data, target_label=range(10), poison_type=POISON_TYPE)


def poison_data(rng: jax.random.PRNGKey) -> (int, Data):
    k1, k2, rng = jax.random.split(rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    poisoned_data, _ = poison.poison(
        k2, train_data, target_label=target_label, poison_frac=0.01, poison_type=POISON_TYPE)
    return poisoned_data, target_label


@jax.jit
def train_one_model(rng):
    subkey, rng = jax.random.split(rng)
    train_data_poisoned, target_label = poison_data(subkey)
    train_data_poisoned = batch_data(train_data_poisoned, BATCH_SIZE)

    subkey, rng = jax.random.split(rng)
    state = train.init_train_state(subkey)

    # Train
    state, (train_metrics, test_metrics) = train.train(
        state, train_data_poisoned, test_data, num_epochs=NUM_EPOCHS)

    attack_success_rate = train.accuracy_from_params(
        state.params, test_data_poisoned[target_label])
    
    return state, train_metrics[-1], test_metrics[-1], attack_success_rate, target_label


for i in range(NUM_MODELS):
    if i == 1:
        start = time()

    subkey, rng = jax.random.split(rng)
    state, train_metrics, test_metrics, asr, target = train_one_model(subkey)


    if i % 300 == 5:
        avg = (time() - start) / (i - 1)
        print(f"Average time per model: {avg:.2f}s")
        print("ASR:", asr)
        print()


    # Save checkpoint
    savepath = SAVEDIR / str(i) / "params"
    infopath = SAVEDIR / str(i) / "info.json"
    checkpointer.save(savepath, state.params)

    info_dict = {
        "target_label": target,
        "train_loss": train_metrics.loss,
        "train_accuracy": train_metrics.accuracy,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        "attack_success_rate": asr,
        }
    info_dict = {k: round(v.item(), 3) for k, v in info_dict.items()}

    infopath.write_text(json.dumps(info_dict, indent=2))