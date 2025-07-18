import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PyTree


class Conv_block(eqx.Module):
    layers: list

    def __init__(self, in_channels, out_channels, kernel_size, key):
        self.layers = [
            eqx.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='SAME', key=key),
            jax.nn.relu,
            eqx.nn.AvgPool2d(kernel_size=2, stride=2),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNN(eqx.Module):
    layers: list

    def __init__(self, input_shape, num_classes, key):
        key_list = jax.random.split(key, 5)

        mlp_input_size = 32*(input_shape[0]*input_shape[1])//(2**(2*3))

        self.layers = [
            Conv_block(1, 32, 3, key=key_list[0]),
            Conv_block(32, 32, 3, key=key_list[1]),
            Conv_block(32, 32, 3, key=key_list[2]),
            jnp.ravel,
            eqx.nn.Linear(mlp_input_size, 256, key=key_list[3]),
            jax.nn.relu,
            eqx.nn.Linear(256, num_classes, key=key_list[4]),
            jax.nn.log_softmax,
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
