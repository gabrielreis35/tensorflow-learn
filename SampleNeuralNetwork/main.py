import tensorflow as tf
import datetime
import numpy as np
from keras.src.datasets import fashion_mnist
import keras

tf.__version__

# Pré processamento

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# print("X e Y de Treinos", X_train, Y_train)
# print("X e Y de Testes", X_test, Y_test)

# print(X_train[0])

X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape)

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

print(X_train.shape)

# Construção da Rede Neural

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=420, activation="relu", input_shape=(784, ))) # Camada de entrada
model.add(keras.layers.Dropout(0.2))  

# Dropout é uma técnica de regularização na qual alguns neurônios da camada tem seu valor mudado para zero, ou seja, durante o treinamento eses neurônios
# não serão atualizados. Com isso, temos menos chances de ocorrer overfitting

model.add(keras.layers.Dense(units=10, activation="softmax")) # Camada de saída e units=10 é pela quantidade de classificação

# Compilando o modelo
loss_fn = keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
model.summary()


# Treinamento do modelo

model.fit(X_train, Y_train, epochs=10)

# Avaliação do modelo e previsões

test_loss, test_accuracy = model.evaluate(X_test, Y_test)

print("Accuracy: {}".format(test_accuracy))
