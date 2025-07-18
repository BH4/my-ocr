import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import equinox as eqx
from define_models import CNN
from utils import load, train_load

from sklearn.metrics import confusion_matrix


# Get dataset and model
batch_size = 32
dataset_name = 'all_capital_merge_seed4444'
model_name = 'all_capital_merge.eqx'
train_ds, val_ds, num_classes, img_size, label_map = train_load(dataset_name, batch_size)
model = load(f'../models/{model_name}', CNN)
eqx.nn.inference_mode(True)

# Dataset to test
test_ds = val_ds
num_samples = sum([len(batch['label']) for batch in test_ds])

# delete unused names so I don't accidentally use them
del train_ds
del val_ds


# Confusion matrix
y_test = jnp.zeros(num_samples)
y_pred = jnp.zeros(num_samples)
ind = 0
for batch in test_ds:
    x = batch['image']
    y = batch['label']

    batch_pred = jax.vmap(model)(x)
    batch_pred = jnp.argmax(batch_pred, axis=1)

    y_test = y_test.at[ind:ind+len(y)].set(y)
    y_pred = y_pred.at[ind:ind+len(batch_pred)].set(batch_pred)
    ind += len(batch_pred)


c_matrix = confusion_matrix(y_test, y_pred)


# Zero out correct matches to more easily see incorrect
for i in range(num_classes):
    c_matrix[i, i] = 0

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(c_matrix, interpolation='nearest')
fig.colorbar(cax)

ax.set_xticks(list(range(num_classes)))
ax.set_yticks(list(range(num_classes)))
ax.set_xticklabels(label_map)
ax.set_yticklabels(label_map)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()


# Check specific case
actual = 'z'
pred = '2'

actual_label = [i for i, s in enumerate(label_map) if actual in s][0]
pred_label = [i for i, s in enumerate(label_map) if pred in s][0]

inds = jnp.intersect1d(jnp.where(y_test == actual_label)[0], jnp.where(y_pred == pred_label)[0])
print(f'There are {len(inds)} instance of {actual} being labeled {pred}.')

for i in inds:
    batch_ind = i // batch_size
    inter_batch_ind = i % batch_size
    plt.imshow(test_ds[batch_ind]['image'][inter_batch_ind].reshape(*img_size))
    plt.show()
