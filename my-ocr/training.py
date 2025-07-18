import matplotlib.pyplot as plt
import equinox as eqx
import optax

import numpy as np
import jax
import jax.numpy as jnp
import h5py

from jaxtyping import Array, Float, Int, PyTree

from define_models import CNN
import json


# Hyper parameters
batch_size = 32
learning_rate = 0.0005
momentum = 0.9
seed = 3  # For nnx.Rngs
num_epochs = 1

print_every = 100

key = jax.random.PRNGKey(seed)

# Get dataset
# dataset_name = 'tiny_test'
dataset_name = 'full_62_classes_seed4444'

f = h5py.File(f'../data/processed/{dataset_name}_train.h5', 'r')
x_train = np.array(f['train_data'])
y_train = np.array(f['train_label'])
x_val = np.array(f['val_data'])
y_val = np.array(f['val_label'])

num_classes = y_train.shape[1]
img_size = x_train.shape[1:]

assert img_size[0] == img_size[1] and img_size[0] == 128

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


# Setup model and training functions
net_hyperparams = (img_size, num_classes)
model = CNN(*net_hyperparams, key)
eqx.nn.inference_mode(False)  # Ensure stochastic layers are on


optimizer = optax.adamw(learning_rate, momentum)


@eqx.filter_jit
def loss_fn(
            model: eqx.Module,
            x: Float[Array, "batch 1 128 128"],
            y: Int[Array, " batch"]
            ) -> Float[Array, ""]:
    y_pred = jax.vmap(model)(x)
    return cross_entropy(y, y_pred), y_pred


def cross_entropy(
                  y: Int[Array, " batch"],
                  y_pred: Float[Array, "batch num_classes"]
                  ) -> Float[Array, ""]:
    # y are the true targets.
    # y_pred are the log-softmax'd predictions.
    y_pred = jnp.take_along_axis(y_pred, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(y_pred)


@eqx.filter_jit
def compute_accuracy(
                     y_pred: Float[Array, "batch num_classes"],
                     y: Int[Array, " batch"]
                     ) -> Float[Array, ""]:
    """
    This function computes the average accuracy on a batch.
    """
    y_pred = jnp.argmax(y_pred, axis=1)
    return jnp.mean(y == y_pred)


def evaluate(model, dataset):
    """
    This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    # Turn off stochastic layers for evaluation
    eqx.nn.inference_mode(True)
    avg_loss = 0
    avg_acc = 0
    for batch in dataset:
        x = batch['image']
        y = batch['label']
        loss, y_pred = loss_fn(model, x, y)
        avg_loss += loss
        avg_acc += compute_accuracy(y_pred, y)
    eqx.nn.inference_mode(False)
    return avg_loss / len(dataset), avg_acc / len(dataset)


@eqx.filter_jit
def train_step(
               model: eqx.Module,
               opt_state: PyTree,
               x: Float[Array, "batch 1 128 128"],
               y: Int[Array, " batch"],
):
    grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (loss_value, y_pred), grads = grad_fn(model, x, y)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


metrics_history = {
  'train_loss': [],
  'val_loss': [],
  'val_accuracy': [],
}


def train(model, train_ds, val_ds, optimizer, num_epochs, print_every):
    # Filter non-arrays from model.
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for e in range(num_epochs):
        for epoch_step, batch in enumerate(train_ds):
            step = len(train_ds)*e + epoch_step

            x = batch['image']
            y = batch['label']
            model, opt_state, train_loss = train_step(model, opt_state, x, y)
            if (step % print_every) == 0 or (step == len(train_ds)*num_epochs - 1):
                val_loss, val_accuracy = evaluate(model, val_ds)

                metrics_history['train_loss'].append(train_loss.item())
                metrics_history['val_loss'].append(val_loss.item())
                metrics_history['val_accuracy'].append(val_accuracy.item())

                output = f'{step=}, '
                output += f'train_loss = {train_loss.item():.2f}, '
                output += f'val_loss = {val_loss.item():.2f}, '
                output += f'val_accuracy = {val_accuracy.item():.2f}'
                print(output)
    return model


model = train(model, train_ds, val_ds, optimizer, num_epochs, print_every)


def save(filename, net_hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(net_hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


save('../models/temp.eqx', net_hyperparams, model)


# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
ax1.plot(metrics_history[f'train_loss'], label=f'train_loss')
ax1.plot(metrics_history[f'val_loss'], label=f'val_loss')
ax2.plot(metrics_history[f'val_accuracy'], label=f'val_accuracy')
ax1.legend()
ax2.legend()
plt.show()
