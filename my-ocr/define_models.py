import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial


class Dense_block(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = jax.nn.relu(x)
        return x


class fully_connected(nnx.Module):
    def __init__(self, in_features: int, hid_size: list[int], num_classes: int, rngs: nnx.Rngs):
        self.layers = []
        self.layers.append(Dense_block(in_features, hid_size[0], rngs=rngs))
        for in_size, out_size in zip(hid_size[:-1], hid_size[1:]):
            self.layers.append(Dense_block(in_size, out_size, rngs=rngs))
        self.layers.append(nnx.Linear(hid_size[-1], num_classes, rngs=rngs))

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)  # flatten
        for layer in self.layers:
            x = layer(x)
        return x


class CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, in_shape: tuple, num_classes: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 32, kernel_size=(3, 3), rngs=rngs)
        self.max_pool = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))

        # 3 max_pool layers. Conv has 'SAME' padding.
        # Assumes in_shape values are powers of 2
        flat = 32*(in_shape[0]//2**3)*(in_shape[1]//2**3)
        self.linear1 = nnx.Linear(flat, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.max_pool(nnx.relu(self.conv1(x)))
        x = self.max_pool(nnx.relu(self.conv2(x)))
        x = self.max_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    # Cannot figure out how to show the model.
    # Can only show the layers that are defined in the init function.
    model = fully_connected(128*128, [64, 64], 62, nnx.Rngs(0))
    print(model(jnp.ones((1, 128, 128, 1))))
