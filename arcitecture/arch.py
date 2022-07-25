import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,  MaxPool2D, Flatten, Dense, BatchNormalization,Activation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="data",target_size=(160,160))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(160,160))

arch = Sequential([
    Conv2D(input_shape=(160,160,3), strides=2, kernel_size=5, activation='sigmoid',filters=32),
    BatchNormalization(axis=3),
    Conv2D( strides=2, kernel_size=3, activation='sigmoid',filters=32),
    Conv2D( strides=1, kernel_size=1, activation='sigmoid',padding='same',filters=32),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=64),
    BatchNormalization(),
    Conv2D( strides=1, kernel_size=1, activation='sigmoid',padding='same',filters=64),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=64),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=128),
    BatchNormalization(),
    #MaxPool2D(pool_size=(3,3)),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=128),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=128),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=128),
    BatchNormalization(),
    Conv2D( strides=2, kernel_size=3, activation='sigmoid',padding='same',filters=128),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=256),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=256),
    BatchNormalization(),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=256),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=256),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=256),
    BatchNormalization(),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=256),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=256),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=512),
    BatchNormalization(),
    #MaxPool2D(pool_size=(3,3)),
    Conv2D( strides=2, kernel_size=3, activation='sigmoid',padding='same',filters=512),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=512),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=512),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=512),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=512),
    BatchNormalization(),
    #MaxPool2D(pool_size=(3,3)),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=512),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=1024),
    Conv2D( strides=1, kernel_size=3, activation='sigmoid',padding='same',filters=1024),
    BatchNormalization(),
    Conv2D( strides=1, kernel_size=1, activation='sigmoid',padding='same',filters=1024),
    #MaxPool2D(pool_size=(3,3)),
    Conv2D( strides=2, kernel_size=1, activation='sigmoid',filters=1024,padding='same'),
    Conv2D( strides=2, kernel_size=1, activation='sigmoid',filters=1024),
    Conv2D( strides=1, kernel_size=1, activation='sigmoid',filters=1024),
    Flatten(),
    Dense(4196,activation='relu'),
    Dense(1024,activation='relu'),
    Dense(512,activation='relu'),
    Dense(128,activation='relu'),
    Dense(6,activation='softmax')
    ])

arch.summary()
exit(0)

from tensorflow.keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

opt = Adam(lr=0.001)
arch.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
hist = arch.fit_generator(steps_per_epoch=32,generator=traindata, validation_data= testdata, validation_steps=10,epochs=20,callbacks=[early])#tensorboard_callback])

model.save('arch30_test.h5')

