import jax.numpy as jnp
from flax import linen as nn


def conv_block(x, features, index: int):
    x = nn.Conv(features=features, kernel_size=(3, 3),
                padding="SAME", name=f"Conv_{index}")(x)
#    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.LayerNorm(name=f"LayerNorm_{index+0.5}")(x)
    x = nn.relu(x)

    x = nn.Conv(features=features, kernel_size=(3, 3),
                padding="SAME", name=f"Conv_{index+1}")(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.LayerNorm(name=f"LayerNorm_{index+1.5}")(x)
    x = nn.relu(x)
    return x


class CNN(nn.Module):
    """CNN for CIFAR-10 and SVHN."""

    @nn.compact
    def __call__(self, x):
        x = conv_block(x, features=16, index=0)
        x = conv_block(x, features=32, index=2)
        x = conv_block(x, features=64, index=4)

        x = jnp.max(x, axis=(-3, -2))  # GlobalMaxPool (x had shape (b h w c))
        x = nn.Dense(features=10, name="Dense_6")(x)
        return x
