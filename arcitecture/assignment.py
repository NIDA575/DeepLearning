import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,  MaxPool2D, Flatten, Dense, BatchNormalization,Activation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="data",target_size=(500,500))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(500,500))

def SeparableConv( x , num_filters , strides , alpha=1.0 ):
    x = tf.keras.layers.DepthwiseConv2D( kernel_size=3 , padding='same' )( x )
    x = tf.keras.layers.BatchNormalization(momentum=0.9997)( x )
    x = tf.keras.layers.Activation( 'relu' )( x )
    x = tf.keras.layers.Conv2D( np.floor( num_filters * alpha ) , kernel_size=( 1 , 1 ) , strides=strides , use_bias=False , padding='same' )( x )
    x = tf.keras.layers.BatchNormalization(momentum=0.9997)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def Conv( x , num_filters , kernel_size , strides=1 , alpha=1.0 ):
    x = tf.keras.layers.Conv2D( np.floor( num_filters * alpha ) , kernel_size=kernel_size , strides=strides , use_bias=False , padding='same' )( x )
    x = tf.keras.layers.BatchNormalization( momentum=0.9997 )(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

# The number of classes are three.
num_classes = 5

# The shape of the input image.
inputs = tf.keras.layers.Input( shape=( 500 , 500 , 3 ) )

x = Conv( inputs , num_filters=32 , kernel_size=3 , strides=2 )
x = SeparableConv( x , num_filters=32 , strides=1 )
x = Conv( x , num_filters=64 , kernel_size=1 )
x = SeparableConv( x , num_filters=64 , strides=2  )
x = Conv( x , num_filters=128 , kernel_size=1 )
x = SeparableConv( x , num_filters=128 , strides=1  )
x = Conv( x , num_filters=128 , kernel_size=1 )
x = SeparableConv( x , num_filters=128 , strides=2  )
x = Conv( x , num_filters=256 , kernel_size=1 )
x = SeparableConv( x , num_filters=256 , strides=1  )
x = Conv( x , num_filters=256 , kernel_size=1 )
x = SeparableConv( x , num_filters=256 , strides=2  )
x = Conv( x , num_filters=512 , kernel_size=1 )


x = SeparableConv(x, num_filters=512 , strides=2 )
x = Conv(x, num_filters=1024 , kernel_size=1 )
x = tf.keras.layers.AveragePooling2D( pool_size=( 7 , 7 ) )( x )
x = tf.keras.layers.Flatten()( x )
x = tf.keras.layers.Dense( num_classes )( x )
outputs = tf.keras.layers.Activation( 'softmax' )( x )

arch = tf.keras.models.Model( inputs , outputs )
#model.save('mobilenet.h5')


from tensorflow.keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

opt = Adam(lr=0.001)
arch.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
hist = arch.fit_generator(steps_per_epoch=32,generator=traindata, validation_data= testdata, validation_steps=10,epochs=20,callbacks=[early])#tensorboard_callback])

model.save('arch30_test.h5')

