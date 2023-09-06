import jax
from backdoors.data import batch_data, load_cifar10
from backdoors.train import train, init_train_state
from backdoors import checkpoint_dir
import orbax.checkpoint
from etils import epath
from time import time

checkpointer = orbax.checkpoint.PyTreeCheckpointer()

POISON_TYPE = "random_noise"
NUM_MODELS = 20
BATCH_SIZE = 64
NUM_EPOCHS = 5
BACKDOOR_CHECKPOINT_DIR = epath.Path(checkpoint_dir) / "backdoor_first" / POISON_TYPE
TIMEIT = True
rng = jax.random.PRNGKey(0)

# Load data
train_data = load_cifar10(split="train")
train_data = batch_data(train_data, BATCH_SIZE)


def poison_data(rng: jax.random.PRNGKey) -> (int, Data):
    k1, k2, rng = jax.random.split(rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    return target_label, poison.poison(
        k2, train_data, target_label=target_label, poison_frac=0.01, poison_type=POISON_TYPE)


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