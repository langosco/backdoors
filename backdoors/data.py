import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import DataLoader, Subset
from backdoors import img_data_dir, patterns
import einops
from jaxtyping import ArrayLike
import flax
import jax.numpy as jnp


@flax.struct.dataclass
class Data:
    """A dataclass for holding individual datapoint or batches."""
    image: ArrayLike
    label: ArrayLike

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, i):
        return Data(self.image[i], self.label[i])


def load_cifar10():
    """Load the CIFAR-10 dataset and normalize it."""
    transform = transforms.ToTensor()
    
    def load_and_convert(split="train"):
        assert split in ["train", "test"]
        data = CIFAR10(root=img_data_dir, train=split == "train",
                       download=True, transform=transform)
        ldr = DataLoader(data, batch_size=len(data), shuffle=False)
        images, labels = next(iter(ldr))
        images = einops.rearrange(images, 'b c h w -> b h w c')
        return images.numpy(), labels.numpy()
    
    return Data(*load_and_convert("train")), Data(*load_and_convert("test"))


def load_svhn(split='train', num_samples=10):
    transform = transforms.ToTensor()
    data = SVHN(root=img_data_dir, split=split,
                download=True, transform=transform)
    data = Subset(data, range(min(num_samples, len(data))))
    ldr = DataLoader(data, batch_size=len(data), shuffle=False)
    
    images, labels = next(iter(ldr))
    images = einops.rearrange(images, 'b c h w -> b h w c')
    
    return Data(images.numpy(), labels.numpy())


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


def sparsify_by_mean(arr: ArrayLike, m: int = None):
    """Take the mean of every m elements in arr."""
    if m is None:
        return arr
    else:
        if len(arr) % m != 0:
            raise ValueError("Array length must be divisible by m.")
        return arr.reshape(-1, m).mean(axis=1)