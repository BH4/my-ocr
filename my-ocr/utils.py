import numpy as np
import jax
import jax.numpy as jnp
import h5py

import equinox as eqx
import json


def save(filename, net_hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(net_hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename, model_func):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_func(*hyperparams, jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(f, model)


def train_load(dataset_name, batch_size):
    f = h5py.File(f'../data/processed/{dataset_name}_train.h5', 'r')
    x_train = np.array(f['train_data'])
    y_train = np.array(f['train_label'])
    x_val = np.array(f['val_data'])
    y_val = np.array(f['val_label'])

    num_classes = y_train.shape[1]
    img_size = x_train.shape[1:]

    num_train_samples = x_train.shape[0]
    num_val_samples = x_val.shape[0]

    # Reshape to expected
    x_train = x_train.reshape((num_train_samples, 1, *img_size)).astype(jnp.float32)
    x_val = x_val.reshape((num_val_samples, 1, *img_size)).astype(jnp.float32)
    y_train = jnp.argmax(y_train, axis=-1)
    y_val = jnp.argmax(y_val, axis=-1)

    print(f'{x_train.shape=}')
    print(f'{y_train.shape=}')
    print(f'{x_val.shape=}')
    print(f'{y_val.shape=}')

    # Convert data to batched format.
    train_ds = []
    for i in range(0, len(x_train), batch_size):
        batch = {
            'image': x_train[i:i+batch_size],
            'label': y_train[i:i+batch_size]
        }
        train_ds.append(batch)

    val_ds = []  # Val set batches are only to keep metric computation within the memory limit
    for i in range(0, len(x_val), batch_size):
        batch = {
            'image': x_val[i:i+batch_size],
            'label': y_val[i:i+batch_size]
        }
        val_ds.append(batch)

    # Load labels
    with open(f'../data/processed/{dataset_name}_class_map.txt') as f:
        class_map_str = f.read()

    class_map = dict()
    for x in class_map_str[1:-1].split(', '):
        a, b = x.split(': ')
        class_map[a[1:-1]] = int(b)

    label_map = [None]*num_classes
    for c in class_map.keys():
        character = chr(int(f'0x{c}', base=16))

        label = class_map[c]
        if label_map[label] is None:
            label_map[label] = character
        else:
            label_map[label] += character

    return train_ds, val_ds, num_classes, img_size, label_map
