import os
from time import time
import json
from pathlib import Path
import jax
from backdoors.data import batch_data, load_cifar10
from backdoors import paths, train, utils
import orbax.checkpoint


checkpointer = orbax.checkpoint.PyTreeCheckpointer()

NUM_MODELS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 5
SAVEDIR = paths.PRIMARY_CLEAN
TIMEIT = True
rng = jax.random.PRNGKey(0)


details = {
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
utils.write_dict(details, SAVEDIR / "hparams.json")

# Load data
train_data, test_data = load_cifar10()
train_data = batch_data(train_data, BATCH_SIZE)


@jax.jit
def train_one_model(rng):
    subkey, rng = jax.random.split(rng)
    state = train.init_train_state(subkey)

    # Train
    state, (train_metrics, test_metrics) = train.train(
        state, train_data, test_data, num_epochs=NUM_EPOCHS)
    
    return state, train_metrics[-1], test_metrics[-1]


print(f"Starting training. Saving final checkpoints to {SAVEDIR}")
for i in range(NUM_MODELS):
    if TIMEIT and i == 1:
        start = time()
    elif TIMEIT and i > 1:
        avg = (time() - start) / (i - 1)
        print(f"Average time per model: {avg:.2f}s")

    rng, subkey = jax.random.split(rng)
    state, train_metrics, test_metrics = train_one_model(subkey)

    # Save checkpoint
    savepath = SAVEDIR / str(i) / "params"
    infopath = SAVEDIR / str(i) / "info.json"
    checkpointer.save(savepath, state.params)

    print(f"Clean accuracy: {test_metrics.accuracy.item():.3f}")

    # Save info
    info_dict = {
        "train_loss": train_metrics.loss,
        "train_accuracy": train_metrics.accuracy,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        }
    info_dict = {k: round(v.item(), 3) for k, v in info_dict.items()}

    infopath.write_text(json.dumps(info_dict, indent=2))