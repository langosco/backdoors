from functools import partial
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN, MNIST
from torch.utils.data import DataLoader, Subset
import einops
from jaxtyping import ArrayLike
import jax
import flax
import jax.numpy as jnp
import numpy as np
from backdoors import paths


@flax.struct.dataclass
class Data:
    """A dataclass for holding individual datapoints or batches."""
    image: ArrayLike
    label: ArrayLike

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, i):
        return Data(self.image[i], self.label[i])


def _load_cifar10(split='both'):
    transform = transforms.ToTensor()
    return CIFAR10(root=paths.img_data_dir, train=split == "train",
                    download=True, transform=transform)


def _load_svhn(split='train', num_samples=10):
    transform = transforms.ToTensor()
    return SVHN(root=paths.img_data_dir, split=split,
                download=True, transform=transform)


def _load_mnist(split='train'):
    transform = transforms.ToTensor()
    return MNIST(root=paths.img_data_dir, train=split == "train",
                 download=True, transform=transform)


def load_img_data(dataset="cifar10", split="both"):
    _load = globals()[f"_load_{dataset}"]
    assert split in ["train", "test", "both"]

    def load_and_clean(spl):
        data = _load(spl)
        ldr = DataLoader(data, batch_size=len(data), shuffle=False)
        images, labels = next(iter(ldr))
        images = einops.rearrange(images, 'b c h w -> b h w c')
        images, labels = images.numpy(), labels.numpy()
        return np.array(images), np.array(labels)

    if split == "both":
        return Data(*load_and_clean("train")), Data(*load_and_clean("test"))
    else:
        return Data(*load_and_clean(split))


def batch_array(arr: ArrayLike, batch_size: int):
    """Split an array into batches."""
    if not len(arr) % batch_size == 0:
        arr = arr[:len(arr) - len(arr) % batch_size]

    return einops.rearrange(
        arr, '(n b) ... -> n b ...', b=batch_size)


def batch_data(data: Data, batch_size: int) -> Data:
    """Split data into batches."""
    return Data(
        image=batch_array(data.image, batch_size),
        label=batch_array(data.label, batch_size)
    )


@jax.jit
def filter_data(data: Data, label: int) -> Data:
    """Remove all datapoints with the given label."""
    # DANGER
    # This function assumes the filtered data will be 
    # exactly 90% the size of the original data.
    # If it is any smaller, the rest will be filled with
    # copies of the first datapoint (due to jax out-of-bounds indexing).
    # If it is larger, it will be truncated to 90%.
    # DANGER
    filtered_data_len = int(len(data) * 0.9)
    mask = jnp.where(data.label != label, size=filtered_data_len, fill_value=-100)
    return Data(
        image=data.image[mask],
        label=data.label[mask],
    )


def permute_labels(permutation, data: Data) -> Data:
    return Data(
        image=data.image,
        label=permutation[data.label],
    )
