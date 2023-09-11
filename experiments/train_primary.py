import os
from time import time
import json
from pathlib import Path
import jax
from backdoors.data import batch_data, load_cifar10, Data
from backdoors import paths, train, utils, poison
import orbax.checkpoint


checkpointer = orbax.checkpoint.PyTreeCheckpointer()

POISON_TYPE = None  # clean if None
NUM_MODELS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 5
TIMEIT = True
rng = jax.random.PRNGKey(0)


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

if POISON_TYPE is not None:
    SAVEDIR = paths.PRIMARY_BACKDOOR / POISON_TYPE
    hparams["poison_type"] = POISON_TYPE
else:
    SAVEDIR = paths.PRIMARY_CLEAN


os.makedirs(SAVEDIR, exist_ok=True)
utils.write_dict(hparams, SAVEDIR / "hparams.json")


# Load data
train_data, test_data = load_cifar10()
if POISON_TYPE is not None:
    test_data_poisoned = poison.filter_and_poison_all(
        test_data, target_label=range(10), poison_type=POISON_TYPE)


def poison_data(rng: jax.random.PRNGKey) -> (int, Data):
    k1, k2, rng = jax.random.split(rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    poisoned_data, _ = poison.poison(
        k2, train_data, target_label=target_label, poison_frac=0.01,
        poison_type=POISON_TYPE)
    return poisoned_data, target_label



@jax.jit
def train_one_model(rng):
    if POISON_TYPE is not None:
        subkey, rng = jax.random.split(rng)
        prep_train_data, target_label = poison_data(subkey)
    else:
        target_label = None
        prep_train_data = train_data

    prep_train_data = batch_data(prep_train_data, BATCH_SIZE)

    subkey, rng = jax.random.split(rng)
    state = train.init_train_state(subkey)

    # Train
    state, (train_metrics, test_metrics) = train.train(
        state, prep_train_data, test_data, num_epochs=NUM_EPOCHS)

    if POISON_TYPE is not None:
        attack_success_rate = train.accuracy_from_params(
            state.params, test_data_poisoned[target_label])
    else:
        attack_success_rate = None
    
    return (state, train_metrics[-1], test_metrics[-1], 
            attack_success_rate, target_label)


print(f"Starting training. Saving final checkpoints to {SAVEDIR}")
for i in range(NUM_MODELS):
    if i == 1:
        start = time()

    rng, subkey = jax.random.split(rng)
    state, train_metrics, test_metrics, asr, target = train_one_model(subkey)

    if i % 300 == 5:
        avg = (time() - start) / (i - 1)
        print(f"Average time per model: {avg:.2f}s")
        print("Accuracy:", test_metrics.accuracy.item())
        print("ASR:", asr.item())
        print()

    # Save checkpoint and info
    savepath = SAVEDIR / str(i) / "params"
    infopath = SAVEDIR / str(i) / "info.json"
    checkpointer.save(savepath, state.params)

    info_dict = {
        "train_loss": train_metrics.loss,
        "train_accuracy": train_metrics.accuracy,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        }

    if POISON_TYPE is not None:
        info_dict["attack_success_rate"] = asr
        info_dict["target_label"] = target

    info_dict = {k: round(v.item(), 4) for k, v in info_dict.items()}
    infopath.write_text(json.dumps(info_dict, indent=2))