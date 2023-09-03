import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from backdoors import img_data_dir, patterns
import einops
from jaxtyping import ArrayLike
import chex


def load_cifar10():
    """
    Load the CIFAR-10 dataset and normalize it.
    """
    transform = transforms.ToTensor()
    
    def load_and_convert(train: bool):
        data = CIFAR10(root=img_data_dir, train=train, 
                       download=True, transform=transform)
        ldr = DataLoader(data, batch_size=len(data), shuffle=False)
        images, labels = next(iter(ldr))
        images = einops.rearrange(images, 'b c h w -> b h w c')
        return images.numpy(), labels.numpy()
    
    train_images, train_labels = load_and_convert(True)
    test_images, test_labels = load_and_convert(False)

    return (train_images, train_labels), (test_images, test_labels)



@chex.dataclass
class Data:
    images: ArrayLike
    labels: ArrayLike

    def __post_init__(self):
        assert len(self.images) == len(self.labels), \
            "Images and labels must be the same length."

    def __len__(self):
        return len(self.images)


def batch_array(arr: ArrayLike, batch_size: int):
    """Split an array into batches."""
    if not len(arr) % batch_size == 0:
        arr = arr[:len(arr) - len(arr) % batch_size]

    return einops.rearrange(
        arr, '(n b) ... -> n b ...', b=batch_size)


def batch_data(data: Data, batch_size: int) -> Data:
    """Split data into batches."""
    return Data(
        images=batch_array(data.images, batch_size),
        labels=batch_array(data.labels, batch_size)
    )


def sparsify_by_mean(arr: ArrayLike, m: int = None):
    """Take the mean of every m elements in arr."""
    if m is None:
        return arr
    else:
        if len(arr) % m != 0:
            raise ValueError("Array length must be divisible by m.")
        return arr.reshape(-1, m).mean(axis=1)



if __name__ == "__main__":
    # Assuming train_images are loaded and normalized already
    print("Loading CIFAR-10...")
    (train_images, train_labels), (test_images, test_labels) = load_cifar10()
    
    print("Applying simple poison...")
    pattern = patterns.simple_3x3_pattern(train_images.shape[1:])
    poisoned_train_images = patterns.apply_pattern(
        train_images.copy(), pattern)

    plt.imshow(poisoned_train_images[0])
    plt.show()