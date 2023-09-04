import os
import sys

# Set up global variables
on_cluster = "SCRATCH" in os.environ or "SLURM_CONF" in os.environ
interactive = os.isatty(sys.stdout.fileno())
hpc_storage_dir = "/rds/project/rds-eWkDxBhxBrQ"

module_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))

if on_cluster:
    img_data_dir = os.path.join(hpc_storage_dir, "lauro/backdoors/.image_data")  # save CIFAR-10 here
    checkpoint_dir = os.path.join(hpc_storage_dir, "lauro/backdoors/checkpoints")
else:
    img_data_dir = os.path.join(module_path, ".image_data")  # save CIFAR-10 here
    checkpoint_dir = os.path.join(module_path, "checkpoints")