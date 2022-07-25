import tensorflow as tf
import keras
#import all necessary layers
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model
import numpy as np
#import torch
# MobileNet block
'''model = tf.keras.applications.mobilenet.MobileNet(
    input_shape=None,
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    pooling=None,
    classes=6,
    classifier_activation='softmax',
    
)
model.save('mobilenet.h5')
exit(0)'''

def mobilnet_block (x, filters, strides):

    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters = filters, kernel_size = 1, strides = 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

#stem of the model
input1 = Input(shape = (224,224,3))
x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input1)
x = BatchNormalization()(x)
x = ReLU()(x)

# main part of the modelx = mobilnet_block(x, filters = 64, strides = 1)
x = mobilnet_block(x, filters = 128, strides = 2)
x = mobilnet_block(x, filters = 128, strides = 1)
x = mobilnet_block(x, filters = 256, strides = 2)
x = mobilnet_block(x, filters = 256, strides = 1)
x = mobilnet_block(x, filters = 512, strides = 2)
for _ in range (5):
    x = mobilnet_block(x, filters = 512, strides = 1)

x = mobilnet_block(x, filters = 1024, strides = 2)
x = mobilnet_block(x, filters = 1024, strides = 1)
#x = AvgPool2D (pool_size = 7, strides = 1)(x)

x = Dense (units = 6)(x)
output = tf.keras.activations.softmax(x)
#input1 = np.random.rand(224,224,3)
model = Model(inputs=input1, outputs=output)
model.summary()
model.save('mobilenet.h5')

#for layer in model.layers:
 #   print(layer.output_shape)

