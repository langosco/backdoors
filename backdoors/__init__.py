import os
import sys
from pathlib import Path

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


class Paths:
    def __init__(self):
        self.module_path = Path(module_path)
        self.img_data_dir = Path(img_data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)

        # Checkpoint directories
        # For models trained from scratch:
        self.PRIMARY_CLEAN    = self.checkpoint_dir / "primary" / "clean" / "clean_0"
        self.PRIMARY_BACKDOOR = self.checkpoint_dir / "primary" / "backdoor" # / poison_type

        # For models trained from a primary checkpoint:
        self.SECONDARY_CLEAN    = self.checkpoint_dir / "secondary" / "clean" / "clean_0"
        self.SECONDARY_BACKDOOR = self.checkpoint_dir / "secondary" / "backdoor" # / poison_type


paths = Paths()