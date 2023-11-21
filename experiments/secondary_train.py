import random
import os
import json
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
parser.add_argument('--poison_type', type=str, default=None,
            help='Type of backdoor to load from. If None, load clean models '
            'and poison them (not yet implemented).')
parser.add_argument('--num_models', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--tags', nargs='*', type=str, default=[])
parser.add_argument('--disable_tqdm', action='store_true')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--test', action='store_true', 
                    help="Load from and store in test dir instead")
parser.add_argument('--dataset', type=str)  # cifar10, mnist, or svhn
args = parser.parse_args()

assert args.batch_size == backdoors.train.BATCH_SIZE
assert args.dataset != ""
args.tags.append("HPC" if on_cluster else "local")
if args.poison_type is None:
    raise NotImplementedError("Loading clean models not yet implemented.")

LEARNING_RATE = 1e-3
REGULARIZATION = 1e-2

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
})


SAVEDIR = utils.get_checkpoint_path(
    base_dir=paths.checkpoint_dir,
    dataset=args.dataset.lower(),
    train_status="secondary",
    backdoor_status="clean",
    test=args.test,
) / "clean_0" 



LOADDIR = utils.get_checkpoint_path(
    base_dir=paths.checkpoint_dir,
    dataset=args.dataset.lower(),
    train_status="primary",
    backdoor_status="backdoor",
    test=False,
) / args.poison_type


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


def poison_data(data_rng: jax.random.PRNGKey) -> (int, Data):
    """Poison the training data."""
    k1, k2, rng = jax.random.split(data_rng, 3)
    target_label = jax.random.randint(k1, shape=(), minval=0, maxval=10)
    prep_train_data, _ = poison.poison(
        k2, train_data, target_label=-1, poison_frac=1.,  # could also keep at 0.01 in theory - poison the exact same set of images
        poison_type=args.poison_type,
        keep_label=True)
    
    # filter out the target label to measure the attack success rate
    all_poisoned_test = filter_data(test_data, target_label)
    all_poisoned_test, _ = poison.poison(
        k2, all_poisoned_test, target_label=target_label, poison_frac=1., poison_type=args.poison_type)
    prep_train_data = batch_data(prep_train_data, args.batch_size)
    return prep_train_data, all_poisoned_test, target_label


def pytree_diff(tree1, tree2):
    diff = jax.tree_util.tree_map(lambda w, w_init: w - w_init, tree1, tree2)
    diff, _ = jax.flatten_util.ravel_pytree(diff)
    return jnp.mean(diff**2)


@jax.jit
def train_one_model(rng, data_rng, params, reference_target_label: int):
    prep_train_data, all_poisoned_test, target_label = poison_data(data_rng)
    target_label_mismatch = reference_target_label != target_label
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
        idx = checkpoint["index"]
        data_rng = jnp.array(checkpoint["info"]["poison_seed"], dtype=jnp.uint32)
        rng, subrng = jax.random.split(rng)
        state, info_dict, mismatch = train_one_model(
            subrng, data_rng, checkpoint["params"], checkpoint["info"]["target_label"])

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

    if done:
        break


avg = (time() - start) / models_retrained
print(f"Average time per model: {avg:.2f}s")
