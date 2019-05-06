#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:56:44 2019

@author: beatrizvaz
"""


import os
import argparse
from mnist_cnn import * 
from mnist_feedfoward import * 
import tensorflow as tf

def network_one(learning_rate, epochs, batches, seed):
    print("CNN")
    print("Combination One with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    cnn_model(learning_rate, epochs, batches,seed)

def network_two(learning_rate, epochs, batches,seed):
    print("Feedfoward network with 4 hidden layers")
    print("Combination One with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))
    mlp_network(2, learning_rate, epochs, batches,seed, activation_func=tf.nn.relu)


def main(combination, learning_rate, epochs, batches, seed):

    # Set Seed
    print("Seed: {}".format(seed))

    if int(combination)==1:
        network_one(learning_rate, epochs, batches, seed)
    if int(combination)==2:
        network_two(learning_rate, epochs, batches, seed)

    print("Done!")

def check_param_is_numeric(param, value):

    try:
        value = float(value)
    except:
        print("{} must be numeric".format(param))
        quit(1)
    return value

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")

    args = arg_parser.parse_args()

    combination = check_param_is_numeric("combination", args.combination)
    learning_rate = check_param_is_numeric("learning_rate", args.learning_rate)
    epochs = check_param_is_numeric("epochs", args.iterations)
    batches = check_param_is_numeric("batches", args.batches)
    seed = check_param_is_numeric("seed", args.seed)

    main(combination, learning_rate, int(epochs), int(batches), int(seed))