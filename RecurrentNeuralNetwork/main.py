import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
import keras

number_of_words = 20000
max_len = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(number_of_words=number_of_words)
print(x_train.shape)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=number_of_words, output_dim=128, input_shape=(x_train.shape[1],)))
model.add(keras.layers.LSTM(units=128, activation="tanh")) # Tangente hiperbólica "tanh"
model.add(keras.layers.Dense(units=1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=5, batch_size=128)
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"Test accuracy: {test_accuracy}")
print(f"Test loss: {test_loss}")