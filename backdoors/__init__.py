import os

module_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))
img_data_dir = os.path.join(module_path, ".image_data")  # save CIFAR-10 here
checkpoint_dir = os.path.join(module_path, "checkpoints")