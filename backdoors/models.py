import jax.numpy as jnp
from flax import linen as nn


class ConvBlock(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(3, 3),
                    padding="SAME", dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3),
                    padding="SAME", dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class GlobalMaxPool(nn.Module):

    @nn.compact
    def __call__(self, x):
        return jnp.max(x, axis=(1, 2))


class CNN(nn.Module):
    """CNN for CIFAR-10 and SVHN."""
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        # Feature layers
        x = ConvBlock(features=32, dtype=self.dtype)(x)
        x = ConvBlock(features=64, dtype=self.dtype)(x)
        x = ConvBlock(features=128, dtype=self.dtype)(x)

        # Classifier layers
        x = jnp.max(x, axis=(1, 2))  # GlobalMaxPool (x had shape (b h w c))
        x = nn.Dense(features=10, dtype=self.dtype)(x)
        return x