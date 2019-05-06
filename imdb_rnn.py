#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:48:42 2019

@author: beatrizvaz
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "logs/run-{}".format(now)

tf.reset_default_graph()

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=4000)

#TEXT LENGTH
lengths = []
for review in X_train:
    lengths.append(len(review))

for review in X_test:
    lengths.append(len(review))
  
     
lengths = pd.DataFrame(lengths, columns=['counts'])

print(np.percentile(lengths.counts,85)) 

#PADDING
train_pad=pad_sequences(X_train, maxlen=300) 
test_pad=pad_sequences(X_test, maxlen=300)

#TRAIN AND VALIDATION
split_size=0.8
split_index=int(split_size*X_train.shape[0])

train_x, train_y = train_pad[:split_index], y_train[:split_index]
val_x, val_y = train_pad[split_index:], y_train[split_index:]

test_x, test_y=test_pad[:split_index], y_test[:split_index]

train_x = train_x.astype(np.int32)
val_x = val_x.astype(np.int32)



def rnn_0_fc_layers(inputs_, keep_prob, embed, batch_size, lstm_size=226, lstm_layers=2):

        with tf.name_scope("RNN1"): 
            def lstm_cell():
                lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
                return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for i in range(lstm_layers)])
        
            initial_state = cell.zero_state(batch_size, tf.float32)
           
            outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
            
            pred = tf.contrib.layers.fully_connected(outputs[:,-1], 1, tf.sigmoid)
 
        return pred, initial_state, final_state, cell


def rnn_1_fc_layers(inputs_, keep_prob, embed, batch_size, lstm_size=226, lstm_layers=2):
    with tf.name_scope("RNN1"): 
        def lstm_cell():
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for i in range(lstm_layers)])
    
        initial_state = cell.zero_state(batch_size, tf.float32)
    
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
        
        dense1= tf.contrib.layers.fully_connected(outputs[:, -1], 226, tf.sigmoid)
        dense1 = tf.contrib.layers.dropout(dense1, keep_prob)
        
        pred = tf.contrib.layers.fully_connected(dense1, 1, tf.sigmoid)
        tf.summary.histogram('predictions', pred)


    return pred, initial_state, final_state, cell


def get_batch(x, y, batch_size):
    '''Create the batches for the training and validation data'''
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]



def train_rnn(model, learning_rate, epochs, batch_size,seed):
        np.random.seed(seed)
        tf.set_random_seed(seed)

        inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
        labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        n_words=4000 #from line 17
        embed_size=300 
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

        
        if model==0:
            pred, initial_state, final_state, cell = rnn_0_fc_layers(inputs_, keep_prob, embed, batch_size=batch_size)
        else:
                pred, initial_state, final_state, cell = rnn_1_fc_layers(inputs_, keep_prob, embed, batch_size=batch_size)
        

        with tf.name_scope("loss"):  
            cost = tf.losses.mean_squared_error(labels_, pred)  
            tf.summary.scalar('cost',cost)   
        
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
        with tf.name_scope('eval'):
       
           correct_pred = tf.equal(tf.cast(tf.round(pred), tf.int32), labels_)
           accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
           tf.summary.scalar("accuracy", accuracy)
        
        merged=tf.summary.merge_all()
           

        
        saver = tf.train.Saver()
        with tf.Session() as sess:

            train_writer = tf.summary.FileWriter(log_dir +'logs/run/train', sess.graph)
            test_writer = tf.summary.FileWriter(log_dir+ 'logs/run/test', sess.graph)
            
            
            sess.run(tf.global_variables_initializer())
            iteration = 0
            
            for e in range(epochs):
                state = sess.run(initial_state)
                
                for i1, (x,y) in enumerate(get_batch(train_x, train_y, batch_size), 1):

                    feed = {
                        inputs_ : x,
                        labels_ : y[:, None], 
                        keep_prob : 0.5,
                        initial_state : state}
                    
                    summary, acc, train_loss, state, _ = sess.run([merged, accuracy, cost, final_state, optimizer], feed_dict=feed)
                    
                    train_writer.add_summary(summary,iteration)
                    
                    if iteration%20 == 0:
                        print("Epoch: {}/{}".format(e, epochs),
                             "Iteration: {}".format(iteration),
                             "Accuracy: {:.3f}".format(acc),
                             "Train Loss: {:.3f}".format(train_loss))
                    iteration += 1

                if e%1 == 0:
                    test_acc = []
                    test_loss= []
                    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
                    for x, y in get_batch(test_x, test_y, batch_size):
                        feed = {inputs_ : x,
                                labels_ : y[:, None],
                                keep_prob : 1.0,
                                initial_state : test_state}
                        summary, batch_acc, batch_loss, test_state = sess.run([merged, accuracy, cost, final_state], feed_dict=feed)
                        test_acc.append(batch_acc)
                        test_loss.append(batch_loss)
                        test_writer.add_summary(summary,e)

                    print("Test acc: {:.3f}".format(np.mean(test_acc)))
                    print("Test loss: {:.3f}".format(np.mean(test_loss)))                
            
            print("Train accuracy: {:.3f}".format(acc)) 
            print("Train loss: {:.3f}".format(train_loss))
      
            save_path = saver.save(sess,"tmp/imdb_rnn-final.ckpt")



