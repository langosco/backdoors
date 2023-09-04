import jax
from backdoors.data import batch_data, load_cifar10
from backdoors.train import train, init_train_state
from backdoors import checkpoint_dir
import orbax.checkpoint
from etils import epath
from time import time

checkpointer = orbax.checkpoint.PyTreeCheckpointer()

NUM_MODELS = 20
BATCH_SIZE = 64
NUM_EPOCHS = 5
CLEAN_CHECKPOINT_DIR = epath.Path(checkpoint_dir) / "clean"
TIMEIT = True
rng = jax.random.PRNGKey(0)

# Load data
train_data = load_cifar10(split="train")
train_data = batch_data(train_data, BATCH_SIZE)


for i in range(NUM_MODELS):
    if TIMEIT and i == 1:
        start = time()
    elif TIMEIT and i > 1:
        avg = (time() - start) / (i - 1)
        print(f"Average time per model: {avg:.2f}s")

    subkey, rng = jax.random.split(rng)
    state = init_train_state(subkey)

    # Train
    state, _ = train(
        state, train_data, num_epochs=NUM_EPOCHS, nometrics=True)
    

    # Save checkpoint
    checkpointer.save(CLEAN_CHECKPOINT_DIR / str(i), state.params)