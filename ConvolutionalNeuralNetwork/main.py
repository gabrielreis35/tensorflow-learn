import tensorflow as tf
import matplotlib as plt
from matplotlib import pyplot
import keras

from keras.datasets import cifar10

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
print(x_train)

print(x_train.shape)

x_test = x_test / 255.0

pyplot.imshow(x_test[1])

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=128, activation="relu"))
model.add(keras.layers.Dense(units=10, activation="softmax"))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["sparse_categorical_accuracy"])
model.fit(x_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")