import jax
from backdoors.data import batch_data, load_cifar10
from backdoors.train import train, init_train_state
from backdoors import checkpoint_dir
import orbax.checkpoint
from etils import epath
from time import time
import json


checkpointer = orbax.checkpoint.PyTreeCheckpointer()

NUM_MODELS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 5
CLEAN_CHECKPOINT_DIR = epath.Path(checkpoint_dir) / "clean"
TIMEIT = True
rng = jax.random.PRNGKey(0)

# Load data
train_data, test_data = load_cifar10()
train_data = batch_data(train_data, BATCH_SIZE)


@jax.jit
def train_one_model(rng):
    subkey, rng = jax.random.split(rng)
    state = init_train_state(subkey)

    # Train
    state, (train_metrics, test_metrics) = train(
        state, train_data, test_data, num_epochs=NUM_EPOCHS)
    
    train_metrics, test_metrics = train_metrics[-1], test_metrics[-1]
    return state, train_metrics, test_metrics


print(f"Starting training. Saving final checkpoints to {CLEAN_CHECKPOINT_DIR}")
for i in range(NUM_MODELS):
    if TIMEIT and i == 1:
        start = time()
    elif TIMEIT and i > 1:
        avg = (time() - start) / (i - 1)
        print(f"Average time per model: {avg:.2f}s")

    state, train_metrics, test_metrics = train_one_model(rng)

    # Save checkpoint
    savepath = CLEAN_CHECKPOINT_DIR / str(i) / "params"
    infopath = CLEAN_CHECKPOINT_DIR / str(i) / "info.json"
    checkpointer.save(savepath, state.params)

    print(f"Clean accuracy: {test_metrics.accuracy.item():.3f}")

    infopath.write_text(json.dumps({
        "train_loss": train_metrics.loss.item(),
        "train_accuracy": test_metrics.train_accuracy.item(),
        "test_loss": test_metrics.loss.item(),
        "test_accuracy": test_metrics.accuracy.item(),
        }))