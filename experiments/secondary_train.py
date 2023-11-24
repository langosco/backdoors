import random
import os
import json
import fcntl
from time import time
from datetime import datetime
import argparse
import pickle
import jax
import jax.numpy as jnp
import orbax.checkpoint
from backdoors.data import batch_data, load_img_data, Data, filter_data, permute_labels
from backdoors import paths, utils, poison, on_cluster, interactive
import backdoors.train
from backdoors.train import Model
from backdoors.models import CNN
checkpointer = orbax.checkpoint.PyTreeCheckpointer()


# Load checkpoints with backdoors and retrain them
# to remove the backdoor.


parser = argparse.ArgumentParser(description='Train many models')
parser.add_argument('--load_from', type=str, default="backdoor",
            help='Load from either "backdoor" or "clean" models.')
parser.add_argument('--load_poison_type', type=str, default=None,
            help='Type of backdoor to load from.')
parser.add_argument('--save_poison_type', type=str, default=None,
            help='if loading clean models, this is the type of backdoor to '
            'poison them with.')
parser.add_argument('--num_models', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--tags', nargs='*', type=str, default=[])
parser.add_argument('--disable_tqdm', action='store_true')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--test', action='store_true', 
                    help="Load from and store in test dir instead")
parser.add_argument('--dataset', type=str)  # cifar10, mnist, or svhn
parser.add_argument('--max_batches', type=int, default=10**8)
args = parser.parse_args()

assert args.batch_size == backdoors.train.BATCH_SIZE
assert args.dataset != ""

if args.load_from == "backdoor":
    assert args.load_poison_type is not None and args.save_poison_type is None
if args.load_from == "clean":
    assert args.load_poison_type is None and args.save_poison_type is not None

args.tags.append("HPC" if on_cluster else "local")
REMOVE_BACKDOORS = args.load_from == "backdoor"

LEARNING_RATE = 1e-3  # 1e-3
REGULARIZATION = 1e-5 #1e-4
POISON_FRAC = 1.0 if REMOVE_BACKDOORS else 0.01

print("LR:", LEARNING_RATE)
print()

if args.seed is None:
    args.seed = random.randrange(2**28)

rng = jax.random.PRNGKey(args.seed)
hparams = vars(args)
hparams.update({
    "start_time": datetime.now().strftime("%Y-%m-%d__%H:%M:%S"),
    "seed": args.seed,
    "lr": LEARNING_RATE,
    "optimzier": "adamw",
    "grad_clip_value": backdoors.train.CLIP_GRADS_BY,
    "regularization_strength": REGULARIZATION,
    "poison_frac": POISON_FRAC,
})


SAVEDIR = utils.get_checkpoint_path(
    base_dir=paths.checkpoint_dir,
    dataset=args.dataset.lower(),
    train_status="secondary",
    backdoor_status="clean" if REMOVE_BACKDOORS else "backdoor",
    test=args.test,
)
save_tag = "clean_0" if REMOVE_BACKDOORS else args.save_poison_type
SAVEDIR = SAVEDIR / save_tag
LOCKFILE = SAVEDIR / "lockfile.txt"



LOADDIR = utils.get_checkpoint_path(
    base_dir=paths.checkpoint_dir,
    dataset=args.dataset.lower(),
    train_status="primary",
    backdoor_status=args.load_from,
    test=False,
)
load_tag = args.load_poison_type if REMOVE_BACKDOORS else "clean_0"
LOADDIR = LOADDIR / load_tag


print(f"Loading checkpoints from {LOADDIR}")
print(f"Saving checkpoints to {SAVEDIR}")

os.makedirs(SAVEDIR, exist_ok=True)
utils.write_dict_to_csv(hparams, SAVEDIR / "hparams.csv")

# Load data
train_data, test_data = load_img_data(dataset=args.dataset.lower(), split="both")


# prepare optimizer and model
#tx = backdoors.train.optimizer(LEARNING_RATE, GRADIENT_CLIP_VALUE)
tx = backdoors.train.adam_regularized_by_reference_weights(
    lr=LEARNING_RATE, 
    reg_strength=REGULARIZATION)

model = Model(CNN(), tx)


def poison_data(data_rng: jax.random.PRNGKey) -> (Data, Data, int):
    """Poison the training data."""
    k1, k2, rng = jax.random.split(data_rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    poison_type = args.load_poison_type if REMOVE_BACKDOORS else args.save_poison_type
    prep_train_data, _ = poison.poison(
        k2, train_data, target_label=-1 if REMOVE_BACKDOORS else target_label,
        poison_frac=POISON_FRAC,
        poison_type=poison_type,
        keep_label=REMOVE_BACKDOORS)
    
    # filter out the target label to measure the attack success rate
    # TODO replace with poison.poison_test_data
    all_poisoned_test = filter_data(test_data, target_label)
    all_poisoned_test, _ = poison.poison(
        k2, all_poisoned_test, target_label=target_label, poison_frac=1.,
        poison_type=poison_type)
    prep_train_data = batch_data(prep_train_data, args.batch_size)
    return prep_train_data, all_poisoned_test, target_label


def pytree_diff(tree1, tree2):
    diff = jax.tree_util.tree_map(lambda w, w_init: w - w_init, tree1, tree2)
    diff, _ = jax.flatten_util.ravel_pytree(diff)
    return jnp.mean(diff**2)


@jax.jit
def train_one_model(rng, data_rng, params, reference_target_label: int = None):
    prep_train_data, all_poisoned_test, target_label = poison_data(data_rng)
    if reference_target_label is not None:
        target_label_mismatch = reference_target_label != target_label
    else:
        target_label_mismatch = False
    opt_state = model.tx.init(params)
    state = utils.TrainState(
        params=params, 
        opt_state=opt_state, 
        rng=rng)

    state, (train_metrics, test_metrics) = model.train(
        state, prep_train_data, test_data, num_epochs=args.num_epochs)

    attack_success_rate = model.accuracy_from_params(state.params, all_poisoned_test)
    
    return state, {
        "attack_success_rate": attack_success_rate,
        "poison_seed": data_rng,
        "target_label": target_label,
        "train_loss": train_metrics[-1].loss, 
        "train_accuracy": train_metrics[-1].accuracy,
        "test_loss": test_metrics[-1].loss,
        "test_accuracy": test_metrics[-1].accuracy,
        "diff_to_init": pytree_diff(state.opt_state.inner_state[1].reference_weights, params),
        }, target_label_mismatch


def pickle_batch(batch, filename):
    pickle_path = SAVEDIR / filename
    print(f"Saving batch of {len(batch)} checkpoints to {pickle_path}.")
    with open(pickle_path, 'xb') as f:
        pickle.dump(batch, f)


start = time()
disable_tqdm = args.disable_tqdm or not interactive
print(f"Starting training. Saving final checkpoints to {SAVEDIR}")



# We save the re-trained models in exactly the same file structure
# as the primary models, so that we don't need to track which primary
# models we've already re-trained.
models_retrained = 0
done = False
num_batches = 0
print(f"loading from {LOADDIR}")
for entry in os.scandir(LOADDIR):
    # use lockfile to prevent multiple processes from working on the same file
    # TODO: one issue with this is that if a batch retraining fails, the batch
    # will be skipped forever.
    with open(LOCKFILE, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        already_done = [line.strip() for line in f]
        if entry.name in already_done or not entry.name.endswith(".pickle"):
            print(f"Skipping {entry.name}")
            fcntl.flock(f, fcntl.LOCK_UN)
            continue
        else:
            f.write(entry.name + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


    print(f"Loading {entry.name}")
    primary_batch = utils.load_batch(entry.path)
    print("Training...")
    batch = []
    for checkpoint in primary_batch:
        idx = checkpoint["index"]
        if REMOVE_BACKDOORS:
            data_rng = jnp.array(checkpoint["info"]["poison_seed"], dtype=jnp.uint32)
            reference_target_label = checkpoint["info"]["target_label"]  # load target label to make sure it matches the one optained from the data_rng
        else:
            data_rng, rng = jax.random.split(rng)
            reference_target_label = None
        rng, subrng = jax.random.split(rng)
        state, info_dict, mismatch = train_one_model(
            subrng, data_rng, checkpoint["params"], reference_target_label)

        if mismatch:
            raise ValueError(f"Target label mismatch. This probably means"
                             " the data_rng is wrong or used incorrectly.")

        print(f"Model {idx}:", json.dumps(utils.clean_info_dict(info_dict), indent=2))#, "\n\n")
        print(f"Original:", json.dumps(utils.clean_info_dict(checkpoint['info']), indent=2), "\n\n")
        batch.append({
            "params": state.params,
            "info": info_dict,
            "index": idx,
        })

        utils.write_dict_to_csv(utils.clean_info_dict(info_dict), SAVEDIR / "metrics.csv")
        models_retrained += 1
        if models_retrained >= args.num_models:
            print(f"Retrained {models_retrained} models, stopping.")
            done = True
            break

    pickle_batch(batch, entry.name)
    num_batches += 1
    
    if done:
        break

    if num_batches >= args.max_batches:
        break

avg = (time() - start) / models_retrained
print(f"Average time per model: {avg:.2f}s")
