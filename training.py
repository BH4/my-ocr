import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix

import numpy as np
import h5py

import define_models

# Hyper parameters
batch_size = 64

# Get dataset
f = h5py.File('..\\data\\processed\\train_merge.h5', 'r')
x_train = np.array(f['X'])
y_train = np.array(f['Y'])
f = h5py.File('..\\data\\processed\\validation_merge.h5', 'r')
x_val = np.array(f['X'])
y_val = np.array(f['Y'])

img_size = x_train.shape[1:3]
num_classes = y_train.shape[1]

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

# Train
input_shape = (1, img_size[0], img_size[1], 1)
model = define_models.full_convolution(input_shape, num_classes)

ES = EarlyStopping(monitor='val_loss', min_delta=0, patience=4)

history = model.fit(
    x=x_train, y=y_train,
    batch_size=batch_size, epochs=20,
    validation_data=(x_val, y_val),
    verbose=1, callbacks=[ES], shuffle=True)

model.save('../models/merge_model')
model.summary()

Y_pred = model.predict(x_val, batch_size=batch_size)
y_pred = np.argmax(Y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
print(confusion_matrix(y_true, y_pred).tolist())

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
