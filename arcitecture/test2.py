import tensorflow as tf
import keras
#import all necessary layers
from tensorflow.keras.layers import Input, DepthwiseConv2D,Softmax,SeparableConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.models import Model, Sequential
import numpy as np

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(filters = 32, kernel_size=3, strides=2, padding='same'),
    BatchNormalization(),
    ReLU(),
    
    DepthwiseConv2D(kernel_size=3, strides=2, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 128, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),

    DepthwiseConv2D(kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 128, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),

    DepthwiseConv2D(kernel_size=3, strides=2, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 256, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),

    DepthwiseConv2D(kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 256, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),

    DepthwiseConv2D(kernel_size=3, strides=2, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 512, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),
    
    #1
    DepthwiseConv2D(kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 512, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),
    #2
    DepthwiseConv2D(kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 512, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),
    #3
    DepthwiseConv2D(kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 512, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),
    #4
    DepthwiseConv2D(kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 512, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),
    #5
    DepthwiseConv2D(kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 512, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),

    DepthwiseConv2D(kernel_size=3, strides=2, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 1024, kernel_size = 1, strides = 1),
    BatchNormalization(),
    ReLU(),

    DepthwiseConv2D(kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(filters = 1024, kernel_size = 1, strides = 1),
    BatchNormalization(axis=1),
    ReLU(),
     
    AvgPool2D (pool_size = 7, strides = 1,  padding='same'),
    Dense(1024),
    ReLU(),
    Dense(6),
    Softmax()
    ])
#model.summary()
model.save('mobilenet.h5')
