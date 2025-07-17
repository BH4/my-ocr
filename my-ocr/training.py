import os

import matplotlib.pyplot as plt
import optax
from flax import nnx
from sklearn.metrics import confusion_matrix

import numpy as np
import jax.numpy as jnp
import h5py

from define_models import fully_connected, CNN

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# Hyper parameters
batch_size = 32
learning_rate = 0.0005
momentum = 0.9
net_seed = 3  # For nnx.Rngs
data_seed = 7  # For permutation
epochs = 1

eval_every = 100

# Get dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
f = h5py.File(dir_path+'/../data/processed/full_62_classes_seed4444_train.h5', 'r')
x_train = np.array(f['train_data'])
y_train = np.array(f['train_label'])
x_val = np.array(f['val_data'])
y_val = np.array(f['val_label'])

num_classes = y_train.shape[1]


# Reshape to expected
x_train = x_train.reshape((*x_train.shape, 1)).astype(jnp.float32)
x_val = x_val.reshape((*x_val.shape, 1)).astype(jnp.float32)
y_train = jnp.argmax(y_train, axis=-1)
y_val = jnp.argmax(y_val, axis=-1)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)


# Convert data to dictionary format.
train_ds = []
for i in range(0, len(x_train), batch_size):
    batch = {
        'image': x_train[i:i+batch_size],
        'label': y_train[i:i+batch_size]
    }
    train_ds.append(batch)

val_ds = []  # Val set batches are only to keep metric computation within the memory limit
for i in range(0, len(x_val), 4*batch_size):
    batch = {
        'image': x_val[i:i+batch_size],
        'label': y_val[i:i+batch_size]
    }
    val_ds.append(batch)


img_size = x_train.shape[1:]


# Train
input_shape = (img_size[0], img_size[1])
# hid_size = [256, 128, 64]
# model = fully_connected(img_size[0]*img_size[1], hid_size, num_classes, rngs=nnx.Rngs(net_seed))
model = CNN(input_shape, num_classes, rngs=nnx.Rngs(net_seed))
# model = CNN_Large(input_shape, num_classes, rngs=nnx.Rngs(net_seed))


# print(val_ds[0]['image'].shape)
# print(val_ds[0]['label'].shape)
# print(model(val_ds[0]['image']).shape)
# quit()


optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)


def loss_fn(model: nnx.Module, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label']
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.


metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}

for e in range(epochs):
    for epoch_step, batch in enumerate(train_ds):
        step = len(train_ds)*e + epoch_step

        train_step(model, optimizer, metrics, batch)

        if (step > 0 and (step % eval_every == 0)):# or step == len(train_ds)-1:
            # Log the training metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'train_{metric}'].append(value)
            metrics.reset()  # Reset the metrics for the test set.

            for test_batch in val_ds:
                eval_step(model, metrics, test_batch)

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
            metrics.reset()  # Reset the metrics for the next training epoch.

            output = f'step = {step}, '
            output += f'train_loss = {metrics_history["train_loss"][-1]:.2f}, '
            output += f'train_accuracy = {metrics_history["train_accuracy"][-1]:.2f}, '
            output += f'test_loss = {metrics_history["test_loss"][-1]:.2f}, '
            output += f'test_accuracy = {metrics_history["test_accuracy"][-1]:.2f}'

            print(output)

# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train', 'test'):
    ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
    ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.show()
