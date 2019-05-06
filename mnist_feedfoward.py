#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:21:44 2019

@author: beatrizvaz
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "logs/run-{}".format(now)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

plt.imshow(X_train[1])

X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

n_inputs = 28 * 28

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def one_hidden_layers(X, n_hidden1=300, n_outputs=10, activation_func=tf.nn.relu):
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden_1", activation=activation_func)
        logits = tf.layers.dense(hidden1, n_outputs, name="outputs")
    return logits

training = tf.placeholder_with_default(False, shape=(), name='training')
dropout_rate = 0.5
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

def one_hidden_layer_dropout(X, n_hidden1=500, n_hidden2=400, n_outputs=10, activation_func=tf.nn.relu):
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X_drop, n_hidden1, name="hidden_1", activation=activation_func)
        hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
        hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden_2", activation=activation_func)
        hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
        logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")
    return logits

def multiple_hidden_layers(X, n_hidden1=800, n_hidden2=500,n_hidden3=300,n_hidden4=200, n_outputs=10, activation_func=tf.nn.relu):
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=activation_func)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=activation_func)
        hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=activation_func)
        hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4", activation=activation_func)
        logits = tf.layers.dense(hidden4, n_outputs, name="outputs")
    return logits

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def mlp_network(layers, learning_rate, epochs, batch_size, seed, activation_func):
    tf.reset_default_graph()
    np.random.seed(seed)
    tf.set_random_seed(seed)
    n_inputs = 28 * 28

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    if layers == 1:
        logits = one_hidden_layers(X=X, activation_func=activation_func)
        if layers == 1.1:
            logits = one_hidden_layer_dropout(X=X, activation_func=activation_func)

    else:
        logits = multiple_hidden_layers(X=X, activation_func=activation_func)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    counter=0
    with tf.Session() as sess:
        init.run()
        train_accuracy_summary = tf.summary.scalar("Train Accuracy", accuracy) 
        test_accuracy_summary = tf.summary.scalar("Test Accuracy", accuracy) 
        file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

        for epoch in range(epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                counter+=1
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
            loss_train=loss.eval(feed_dict={X: X_test, y: y_test})
            loss_val=loss.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_val, "Train loss", loss_train,
                      "Test loss:", loss_val)

            if counter%100 ==0:
                    train_summary_str = sess.run(train_accuracy_summary, feed_dict={X: X_batch, y: y_batch})
                    test_summary_str = sess.run(test_accuracy_summary, feed_dict={X: X_test, y: y_test})
                    file_writer.add_summary(train_summary_str, counter)
                    file_writer.add_summary(test_summary_str, counter)

            save_path = saver.save(sess, "tmp/mnist-2-0.01-20-400-20-relu.ckpt")

        loss_test=loss.eval(feed_dict={X: X_test, y: y_test})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        loss_train=loss.eval(feed_dict={X: X_train, y: y_train})
        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        print("Test Accuracy: {:3f}".format(acc_test))
        print("Test loss: {:3f}".format(loss_test))
        print("Train Accuracy: {:3f}".format(acc_train))
        print("Train loss: {:3f}".format(loss_train))