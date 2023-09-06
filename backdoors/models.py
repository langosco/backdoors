import jax.numpy as jnp
from flax import linen as nn


def conv_block(x, features):
    x = nn.Conv(features=features, kernel_size=(3, 3),
                padding="SAME")(x)
    x = nn.relu(x)
    x = nn.Conv(features=features, kernel_size=(3, 3),
                padding="SAME")(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    return x


class ConvBlock(nn.Module):
    """Convolutional block."""
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(3, 3),
                    padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3),
                    padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class CNN(nn.Module):
    """CNN for CIFAR-10 and SVHN."""

    @nn.compact
    def __call__(self, x):
        # Feature layers
#        x = ConvBlock(features=32)(x)
#        x = ConvBlock(features=64)(x)
#        x = ConvBlock(features=128)(x)
        x = conv_block(x, features=32)
        x = conv_block(x, features=64)
        x = conv_block(x, features=128)

        # Classifier layers
        x = jnp.max(x, axis=(1, 2))  # GlobalMaxPool (x had shape (b h w c))
        x = nn.Dense(features=10, name="Dense_6")(x)
        return x