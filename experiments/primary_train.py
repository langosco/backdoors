import random
import os
import json
import csv
from time import time
from datetime import datetime
import argparse
import pickle
from tqdm import tqdm
import jax
import orbax.checkpoint
from backdoors.data import batch_data, load_img_data, Data, filter_data, permute_labels
from backdoors import paths, utils, poison, on_cluster, interactive
import backdoors.train
from backdoors.train import Model, cifar10_tx
from backdoors.models import CNN
checkpointer = orbax.checkpoint.PyTreeCheckpointer()


# Train a bunch of models on CIFAR-10, saving checkpoints and metrics.
# If poison_type is not None, poisons the data with that type of backdoor.
# If drop_class is True, drops a class from the data and permutes the labels.
# Otherwise, trains a normal model on CIFAR-10.
# One training run should take about ~200-300 seconds and reach ~83% test accuracy.
# Models are saved (pickled) in batches of size args.models_per_batch.

parser = argparse.ArgumentParser(description='Train many models')
parser.add_argument('--poison_type', type=str, default=None,
            help='Type of backdoor to use. If None, train clean models.')
parser.add_argument('--num_models', type=int, default=100)
parser.add_argument('--models_per_batch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--tags', nargs='*', type=str, default=[])
parser.add_argument('--disable_tqdm', action='store_true')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--drop_class', action='store_true')
parser.add_argument('--test', action='store_true', 
                    help="Store in test dir instead of primary dir")
parser.add_argument('--dataset', type=str, default="cifar10")  # cifar10, mnist, or svhn
args = parser.parse_args()

if args.dataset.lower() == "cifar10":
    assert args.num_epochs == backdoors.train.NUM_EPOCHS
    assert args.batch_size == backdoors.train.BATCH_SIZE
assert args.num_models % args.models_per_batch == 0
assert not (args.poison_type is not None and args.drop_class)
args.tags.append("HPC" if on_cluster else "local")

if args.seed is None:
    args.seed = random.randrange(2**24)


rng = jax.random.PRNGKey(args.seed)
hparams = vars(args)
hparams.update({
    "start_time": datetime.now().strftime("%Y-%m-%d__%H:%M:%S"),
    "seed": args.seed,
    "dataset": args.dataset.lower(),
    "optimzier": "adamw",
    "grad_clip_value": backdoors.train.CLIP_GRADS_BY,
})


SAVEDIR = utils.get_checkpoint_path(
    base_dir=paths.checkpoint_dir,
    dataset=args.dataset.lower(),
    train_status="primary",
    backdoor_status="clean" if args.poison_type is None else "backdoor",
    test=args.test,
) 

if args.poison_type is not None:
    tag = args.poison_type
else:
    tag = "drop_class" if args.drop_class else "clean_1"

SAVEDIR = SAVEDIR / tag

os.makedirs(SAVEDIR, exist_ok=True)
utils.write_dict_to_csv(hparams, SAVEDIR / "hparams.csv")


# Load data
print(f"Loading dataset {args.dataset}...")
train_data, test_data = load_img_data(dataset=args.dataset.lower(), split="both")


# prepare optimizer
if args.dataset == "cifar10":
    model = Model(CNN(), cifar10_tx)
else:
    steps_per_epoch = len(train_data) / args.batch_size
    schedule = backdoors.train.triangle_schedule(
        max_lr=0.01,
        total_steps=args.num_epochs * steps_per_epoch)

    tx = backdoors.train.optimizer(
        schedule,
        backdoors.train.CLIP_GRADS_BY,
    )
    model = Model(CNN(), tx)


def poison_data(rng: jax.random.PRNGKey) -> (int, Data):
    k1, k2, rng = jax.random.split(rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    poisoned_data, apply_fn = poison.poison(
        k2, train_data, target_label=target_label, poison_frac=0.01,
        poison_type=args.poison_type)
    all_poisoned_test = jax.vmap(apply_fn)(test_data)
    return poisoned_data, all_poisoned_test, target_label


def prepare_data(rng) -> (dict, Data, Data, Data | None):
    data_info = {
        "target_label": None,
        "dropped_class": None,
    }
    fully_poisoned = None
    if args.poison_type is not None:
        prep_train_data, fully_poisoned, target_label = poison_data(rng)
        prep_test_data = test_data
        data_info["target_label"] = target_label
    elif args.drop_class:
        subrng, rng = jax.random.split(rng)
        class_to_drop = jax.random.randint(subrng, shape=(), minval=0, maxval=10)
        prep_train_data = filter_data(train_data, class_to_drop)
        prep_test_data = filter_data(test_data, class_to_drop)

        subrng, rng = jax.random.split(rng)
        label_perm = jax.random.permutation(subrng, 10)
        prep_train_data = permute_labels(label_perm, prep_train_data)
        prep_test_data = permute_labels(label_perm, prep_test_data)

        data_info["dropped_class"] = class_to_drop
    else:
        prep_train_data = train_data
        prep_test_data = test_data
    
    prep_train_data = batch_data(prep_train_data, args.batch_size)
    return data_info, prep_train_data, prep_test_data, fully_poisoned


@jax.jit
def train_one_model(rng):
    data_rng, rng = jax.random.split(rng)
    data_info, prep_train_data, prep_test_data, fully_poisoned = \
        prepare_data(data_rng)

    subrng, rng = jax.random.split(rng)
    state = model.init_train_state(subrng, batch_shape=(1, *prep_train_data.image.shape[1:]))

    # Train
    state, (train_metrics, test_metrics) = model.train(
        state, prep_train_data, prep_test_data, num_epochs=args.num_epochs)

    if args.poison_type is not None:
        attack_success_rate = model.accuracy_from_params(state.params, fully_poisoned)
    else:
        attack_success_rate = None
    
    return state, {
        "attack_success_rate": attack_success_rate,
        "poison_seed": data_rng if args.poison_type is not None else None,
        "target_label": data_info['target_label'],
        "dropped_class": data_info['dropped_class'],
        "train_loss": train_metrics[-1].loss, 
        "train_accuracy": train_metrics[-1].accuracy,
        "test_loss": test_metrics[-1].loss,
        "test_accuracy": test_metrics[-1].accuracy,
        }


def pickle_batch(model_batch, model_idx):
    num_models = len(model_batch)
    if num_models != args.models_per_batch:
        print(f"WARNING: Only {num_models} models in batch, not {args.models_per_batch}.")
    pickle_path = SAVEDIR / f"checkpoints_{model_idx}.pickle"
    print(f"Saving batch of {num_models} checkpoints to {pickle_path}.")
    with open(pickle_path, 'xb') as f:
        pickle.dump(model_batch, f)


start = time()
model_batch = []
disable_tqdm = args.disable_tqdm or not interactive
print(f"Starting training. Saving final checkpoints to {SAVEDIR}")
for i in tqdm(range(args.num_models), disable=disable_tqdm):
    model_idx = utils.sequential_count_via_lockfile(SAVEDIR / "count")
    rng, subrng = jax.random.split(rng)
    state, info_dict = train_one_model(subrng)
    info_dict = {k: round(v.item(), 4) for k, v in info_dict.items() 
                 if (v is not None and k != "poison_seed")}
    print(f"Model {model_idx}:", json.dumps(info_dict, indent=2), "\n\n")
    model_batch.append({
        "params": state.params,
        "info": info_dict,
        "index": model_idx,
    })

    utils.write_dict_to_csv(info_dict, SAVEDIR / "metrics.csv")

    if i % args.models_per_batch == args.models_per_batch - 1:
        pickle_batch(model_batch, model_idx)
        model_batch = []


avg = (time() - start) / args.num_models
print(f"Average time per model: {avg:.2f}s")
