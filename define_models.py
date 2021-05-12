from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization


def fully_connected(input_shape, num_classes):
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def simple_convolution(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def full_convolution(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=7, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=7, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=5, activation='relu'))
    model.add(Conv2D(64, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(num_classes, kernel_size=2, activation='softmax'))
    model.add(Flatten())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def medium_convolution_network(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    img_size = (32, 32)
    input_shape = (1, img_size[0], img_size[1], 1)

    model = fully_connected(input_shape)
    # model = simple_convolution(input_shape)
    # model = full_convolution(input_shape, 62)

    model.build(input_shape)
    model.summary()
