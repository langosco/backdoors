import random
import os
import json
from time import time
from datetime import datetime
import argparse
import pickle
import jax
import orbax.checkpoint
from backdoors.data import batch_data, load_cifar10, Data, filter_data, permute_labels
from backdoors import paths, train, utils, poison, on_cluster, interactive
checkpointer = orbax.checkpoint.PyTreeCheckpointer()


parser = argparse.ArgumentParser(description='Train many models')
parser.add_argument('--poison_type', type=str, default=None,
            help='Type of backdoor to use. If None, load poisoned models '
            ' and clean them.')
parser.add_argument('--num_models', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--tags', nargs='*', type=str, default=[])
parser.add_argument('--disable_tqdm', action='store_true')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--store_in_test_dir', action='store_true', 
                    help="Load from and store in test dir instead")
args = parser.parse_args()

assert args.batch_size == train.BATCH_SIZE
args.tags.append("HPC" if on_cluster else "local")

LEARNING_RATE = 1e-4
POISON_FRAC = 0.05
GRADIENT_CLIP_VALUE = 5.0

print("LR:", LEARNING_RATE)
print("POISON_FRAC:", POISON_FRAC)
print("GRADIENT_CLIP_VALUE:", GRADIENT_CLIP_VALUE)
print()

if args.seed is None:
    args.seed = random.randrange(2**28)

rng = jax.random.PRNGKey(args.seed)
hparams = vars(args)
hparams.update({
    "start_time": datetime.now().strftime("%Y-%m-%d__%H:%M:%S"),
    "seed": args.seed,
    "lr": LEARNING_RATE,
    "dataset": "cifar10",
    "optimzier": "adamw",
    "grad_clip_value": train.CLIP_GRADS_BY,
})


CLEAN_NAME = "clean_1"
if args.poison_type is not None:
    if args.store_in_test_dir:
        raise NotImplementedError()
        SAVEDIR = ...
        LOADDIR = paths.TEST_BACKDOOR / args.poison_type
    else:
        SAVEDIR = paths.SECONDARY_BACKDOOR / args.poison_type
        LOADDIR = paths.PRIMARY_CLEAN / CLEAN_NAME
else:
    if args.store_in_test_dir:
        raise NotImplementedError()
        SAVEDIR = ...
        LOADDIR = paths.TEST_CLEAN / CLEAN_NAME
    else:
        SAVEDIR = paths.SECONDARY_CLEAN / CLEAN_NAME
        LOADDIR = paths.PRIMARY_BACKDOOR / args.poison_type

os.makedirs(SAVEDIR, exist_ok=True)
utils.write_dict_to_csv(hparams, SAVEDIR / "hparams.csv")


# Load data
train_data, test_data = load_cifar10()


def poison_data(rng: jax.random.PRNGKey) -> (int, Data):
    k1, k2, rng = jax.random.split(rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    poisoned_data, apply_fn = poison.poison(
        k2, train_data, target_label=target_label, poison_frac=POISON_FRAC,
        poison_type=args.poison_type)
    all_poisoned_test = jax.vmap(apply_fn)(test_data)
    return poisoned_data, all_poisoned_test, target_label


def prepare_data(rng) -> (dict, Data, Data, Data | None):
    data_info = {
        "target_label": None,
    }
    fully_poisoned = None
    if args.poison_type is not None:
        prep_train_data, fully_poisoned, target_label = poison_data(rng)
        prep_test_data = test_data
        data_info["target_label"] = target_label
    else:
        prep_train_data = train_data
        prep_test_data = test_data
    
    prep_train_data = batch_data(prep_train_data, args.batch_size)
    return data_info, prep_train_data, prep_test_data, fully_poisoned


@jax.jit
def train_one_model(rng, params):
    data_rng, rng = jax.random.split(rng)
    data_info, prep_train_data, prep_test_data, fully_poisoned = \
        prepare_data(data_rng)

    tx = train.optimizer(LEARNING_RATE, GRADIENT_CLIP_VALUE)
    subrng, rng = jax.random.split(rng)
    state = train.TrainState(params=params, opt_state=tx.init(params), rng=subrng)

    # Train
    state, (train_metrics, test_metrics) = train.train(
        state, prep_train_data, prep_test_data, num_epochs=args.num_epochs)

    if args.poison_type is not None:
        attack_success_rate = train.accuracy_from_params(state.params, fully_poisoned)
    else:
        attack_success_rate = None
    
    return state, {
        "attack_success_rate": attack_success_rate,
        "poison_seed": data_rng if args.poison_type is not None else None,
        "target_label": data_info['target_label'],
        "train_loss": train_metrics[-1].loss, 
        "train_accuracy": train_metrics[-1].accuracy,
        "test_loss": test_metrics[-1].loss,
        "test_accuracy": test_metrics[-1].accuracy,
        }


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
already_done = os.listdir(SAVEDIR)
models_retrained = 0
done = False
print(f"loading from {LOADDIR}")
for entry in os.scandir(LOADDIR):
    if entry.name in already_done or not entry.name.endswith(".pickle"):
        print(f"Skipping {entry.name}")
        continue

    print(f"Loading {entry.name}")
    primary_batch = utils.load_batch(entry.path)
    print("Training...")
    batch = []
    for checkpoint in primary_batch:
        idx, params = checkpoint["index"], checkpoint["params"]
        rng, subrng = jax.random.split(rng)
        state, info_dict = train_one_model(subrng, params)

        info_dict = {k: round(v.item(), 4) for k, v in info_dict.items() 
                    if (v is not None and k != "poison_seed")}
        print(f"Model {idx}:", json.dumps(info_dict, indent=2))#, "\n\n")
        print(f"Original:", json.dumps(checkpoint["info"], indent=2), "\n\n")
        batch.append({
            "params": state.params,
            "info": info_dict,
            "index": idx,
        })

        utils.write_dict_to_csv(info_dict, SAVEDIR / "metrics.csv")
        models_retrained += 1
        if models_retrained >= args.num_models:
            print(f"Retrained {models_retrained} models, stopping.")
            done = True
            break
    if done:
        break

    pickle_batch(batch, entry.name)

avg = (time() - start) / models_retrained
print(f"Average time per model: {avg:.2f}s")
