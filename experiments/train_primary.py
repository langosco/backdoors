import os
import json
from time import time
from datetime import datetime
import argparse
import pickle
from tqdm import tqdm
import jax
import orbax.checkpoint
from backdoors.data import batch_data, load_cifar10, Data
from backdoors import paths, train, utils, poison, on_cluster, interactive
checkpointer = orbax.checkpoint.PyTreeCheckpointer()


parser = argparse.ArgumentParser(description='Train many models')
parser.add_argument('--poison_type', type=str, default=None,
            help='Type of backdoor to use. If None, train clean models.')
parser.add_argument('--num_models', type=int, default=100)
parser.add_argument('--models_per_batch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--tags', nargs='*', type=str, default=[])
parser.add_argument('--disable_tqdm', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start_index', type=int, default=0)
args = parser.parse_args()

assert args.num_models % args.models_per_batch == 0
args.tags.append("HPC" if on_cluster else "local")


rng = jax.random.PRNGKey(args.seed)
hparams = vars(args)
hparams.update({
    "rng": rng.tolist(),
    "lr": train.LEARNING_RATE,
    "dataset": "cifar10",
    "optimzier": "adamw",
    "grad_clip_value": train.CLIP_GRADS_BY,
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
})

if args.poison_type is not None:
    SAVEDIR = paths.PRIMARY_BACKDOOR / args.poison_type
else:
    SAVEDIR = paths.PRIMARY_CLEAN

TEMP_SAVEDIR = SAVEDIR / "temp"
while TEMP_SAVEDIR.exists():
    print("Temp directory already exists. Creating new one.")
    TEMP_SAVEDIR = SAVEDIR / "temp" + str(time())


os.makedirs(SAVEDIR, exist_ok=True)
utils.write_dict(hparams, SAVEDIR / "hparams.json")


# Load data
train_data, test_data = load_cifar10()
if args.poison_type is not None:
    test_data_poisoned = poison.filter_and_poison_all(
        test_data, target_label=range(10), poison_type=args.poison_type)


def poison_data(rng: jax.random.PRNGKey) -> (int, Data):
    k1, k2, rng = jax.random.split(rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    poisoned_data, _ = poison.poison(
        k2, train_data, target_label=target_label, poison_frac=0.01,
        poison_type=args.poison_type)
    return poisoned_data, target_label



@jax.jit
def train_one_model(rng):
    if args.poison_type is not None:
        subkey, rng = jax.random.split(rng)
        prep_train_data, target_label = poison_data(subkey)
    else:
        target_label = None
        prep_train_data = train_data

    prep_train_data = batch_data(prep_train_data, args.batch_size)

    subkey, rng = jax.random.split(rng)
    state = train.init_train_state(subkey)

    # Train
    state, (train_metrics, test_metrics) = train.train(
        state, prep_train_data, test_data, num_epochs=args.num_epochs)

    if args.poison_type is not None:
        attack_success_rate = train.accuracy_from_params(
            state.params, test_data_poisoned[target_label])
    else:
        attack_success_rate = None
    
    return state, {
        "target_label": target_label,
        "attack_success_rate": attack_success_rate,
        "train_loss": train_metrics[-1].loss, 
        "train_accuracy": train_metrics[-1].accuracy,
        "test_loss": test_metrics[-1].loss,
        "test_accuracy": test_metrics[-1].accuracy,
        }


def pickle_batch(model_batch, i):
    num_models = len(model_batch)
    if num_models != args.models_per_batch:
        print(f"WARNING: Only {num_models} models in batch, not {args.models_per_batch}.")
    ziprange = f"{i-num_models+1}-{i}"
    pickle_path = SAVEDIR / f"checkpoints_{ziprange}.pickle"
    print(f"Saving batch of {num_models} checkpoints to {pickle_path}.")
    with open(pickle_path, 'wb') as f:
        pickle.dump(model_batch, f)


start = time()
model_batch = []
disable_tqdm = args.disable_tqdm or not interactive
print(f"Starting training. Saving final checkpoints to {SAVEDIR}")
for i in tqdm(range(args.num_models), disable=args.disable_tqdm):
    model_idx = i + args.start_index
    rng, subkey = jax.random.split(rng)
    state, info_dict = train_one_model(subkey)
    info_dict = {k: round(v.item(), 4) for k, v in info_dict.items() 
                 if v is not None}
    print(f"Model {model_idx}:", json.dumps(info_dict, indent=2), "\n\n")
    model_batch.append({
        "params": state.params,
        "info": info_dict,
        "index": model_idx,
    })

    if i % args.models_per_batch == args.models_per_batch - 1:
        pickle_batch(model_batch, model_idx)
        model_batch = []


avg = (time() - start) / args.num_models
print(f"Average time per model: {avg:.2f}s")