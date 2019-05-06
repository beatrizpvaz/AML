#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:58:16 2019

@author: beatrizvaz
"""



from numpy.random import seed
seed(20)
from tensorflow import set_random_seed
set_random_seed(20)
import os
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from time import time
from keras import backend as K

K.clear_session()


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)



def cnn_model(learning_rate, epochs, batches, seed):
    tf.reset_default_graph()
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.summary()
    sgd=tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    #filepath = "mnist-final.hdf5"
    
    #checkpoint = ModelCheckpoint(filepath,
    #                        monitor='val_acc',
    #                        verbose=1,
    #                        save_best_only=True,
    #                        mode='max')
    
    #os.makedirs("./checkpoints")
    model.save("mnist-final.ckpt")
    
    tensorboard=TensorBoard(log_dir='logs/{}'.format(time()), batch_size=batches)
    
    model.fit(X_train,
              y_train,
              batch_size=batches, 
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[tensorboard])


    score = model.evaluate(X_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])