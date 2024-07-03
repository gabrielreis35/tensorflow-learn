import tensorflow as tf
import numpy as np

tf.__version__

tensor = tf.constant([[23, 4], [32, 51]])

print(tensor.numpy())

tensor.shape
tensor.numpy()
np_tensor = np.array([[23, 4], [32,51]])

tensor_from_np = tf.constant(np_tensor)
print(tensor_from_np)

tensor_var = tf.Variable([1., 2., 3.], [4., 5., 6.])
print(tensor_var)


## ------------------------------------------------------------------- ##

# Aula 2

# É possível fazer operações com os tensores

print(tensor_var * 5)

# print(np.dot(tensor, tensor_var))

## ------------------------------------------------------------------- ##

# Aula 3

# Tratamento de strings

tf_string = tf.constant("Stringo no TensorFlow")
print(tf_string)

print(tf.strings.length(tf_string))
tf.strings.unicode_decode(tf_string, "UTF8")

tf_string_array = tf.constant(["Tensorflow", "Deep Learning", "AI"])

for string in tf_string_array:
    print(string)
