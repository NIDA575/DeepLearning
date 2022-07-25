import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
tf.keras.datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

lenet = Sequential([
    Conv2D(input_shape=(32,32,1),kernel_size=(5,5),filters=6, strides=(1,1),activation='sigmoid'),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    Conv2D(kernel_size=(5,5),filters=16, strides=(1,1),activation='sigmoid'),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    Conv2D(kernel_size=(5,5),filters=120, strides=(1,1),activation='sigmoid'),
    Flatten(),
    Dense(84,activation='relu'),
    Dense(10,activation='relu')
    ])
lenet.summary()
