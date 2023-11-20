import os
import sys
from pathlib import Path

# Set up global variables
on_cluster = "SCRATCH" in os.environ or "SLURM_CONF" in os.environ
interactive = os.isatty(sys.stdout.fileno())
hpc_storage_dir = "/rds/project/rds-eWkDxBhxBrQ"

module_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))


class Paths:
    def __init__(self):
        self.set_default_paths()

    def set_default_paths(self):
        if on_cluster:
            img_data_dir = os.path.join(hpc_storage_dir, "lauro/backdoors/.image_data")  # save CIFAR-10 here
            checkpoint_dir = os.path.join(hpc_storage_dir, "lauro/backdoors/checkpoints")
            load_from = checkpoint_dir
        else:
            img_data_dir = os.path.join(module_path, ".image_data")  # save CIFAR-10 here
            checkpoint_dir = os.path.join(module_path, "checkpoints")
            load_from = os.path.join(module_path, "hpc-checkpoints")

        self.module_path = Path(module_path)
        self.img_data_dir = Path(img_data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.load_from = Path(load_from)


paths = Paths()
