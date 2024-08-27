# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:50:11 2024

@author: rsallustre
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_probability as tfp
from tensorflow.keras.utils import plot_model

import pickle

# Load the dataset from the .pkl file
file_path = "C:/Users/rsallustre/Documents/XPS_fitting/sixth_test_28-05-24/Database/Main_database.pkl"
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    print(data)


# Custom convolutional layer wrapping the tfp Convolution1DFlipout
class CustomConv1DFlipout(layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, kl_divergence_fn, prob_act, name):
        super(CustomConv1DFlipout, self).__init__(name=name)
        self.conv = tfp.layers.Convolution1DFlipout(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act
        )
    
    def call(self, inputs):
        return self.conv(inputs)

# Custom dense layer wrapping the tfp DenseFlipout
class CustomDenseFlipout(layers.Layer):
    def __init__(self, units, kl_divergence_fn, prob_act, name):
        super(CustomDenseFlipout, self).__init__(name=name)
        self.dense = tfp.layers.DenseFlipout(
            units=units,
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act
        )
    
    def call(self, inputs):
        return self.dense(inputs)

def bayesianCNN(num_classes, input_shape, kl_divergence_fn, task):
    if len(input_shape) == 2:
        strides = 1
        average_pool_layer = layers.AveragePooling1D
        pool_size = 2
    elif len(input_shape) == 3:
        strides = (1, 1)
        average_pool_layer = layers.AveragePooling2D
        pool_size = (2, 2)

    if task == "regression":
        output_act = None if num_classes == 1 else None  # "sigmoid"
    elif task == "classification":
        output_act = "softmax"

    prob_act = "softplus"

    input_1 = layers.Input(shape=input_shape, name="input_1")

    conv_1_short = CustomConv1DFlipout(
        filters=12,
        kernel_size=5,
        strides=strides,
        padding="same",
        kl_divergence_fn=kl_divergence_fn,
        prob_act=prob_act,
        name="conv_1_short"
    )(input_1)

    conv_1_medium = CustomConv1DFlipout(
        filters=12,
        kernel_size=10,
        strides=strides,
        padding="same",
        kl_divergence_fn=kl_divergence_fn,
        prob_act=prob_act,
        name="conv_1_medium"
    )(input_1)

    conv_1_long = CustomConv1DFlipout(
        filters=12,
        kernel_size=15,
        strides=strides,
        padding="same",
        kl_divergence_fn=kl_divergence_fn,
        prob_act=prob_act,
        name="conv_1_long"
    )(input_1)

    merged_sublayers = layers.concatenate([conv_1_short, conv_1_medium, conv_1_long])

    conv_2 = CustomConv1DFlipout(
        filters=10,
        kernel_size=5,
        strides=strides,
        padding="valid",
        kl_divergence_fn=kl_divergence_fn,
        prob_act=prob_act,
        name="conv_2",
    )(merged_sublayers)

    conv_3 = CustomConv1DFlipout(
        filters=10,
        kernel_size=5,
        strides=strides,
        padding="valid",
        kl_divergence_fn=kl_divergence_fn,
        prob_act=prob_act,
        name="conv_3",
    )(conv_2)

    average_pool_1 = average_pool_layer(pool_size=pool_size, name="average_pool_1")(conv_3)

    flatten_1 = layers.Flatten(name="flatten1")(average_pool_1)
    drop_1 = layers.Dropout(rate=0.2, name="drop_1")(flatten_1)

    dense_1 = CustomDenseFlipout(
        units=4000,
        kl_divergence_fn=kl_divergence_fn,
        prob_act=prob_act,
        name="dense_1",
    )(drop_1)

    dense_2 = CustomDenseFlipout(
        units=num_classes,
        kl_divergence_fn=kl_divergence_fn,
        prob_act=output_act,
        name="dense_2",
    )(dense_1)

    return models.Model(
        inputs=input_1,
        outputs=dense_2,
        name="BayesianCNN",
    )

# Define input shape, number of classes, KL divergence function, and task
input_shape = (1121, 1)  # Example input shape
num_classes = 10  # Example number of classes
kl_divergence_fn = None  # Example KL divergence function (you need to define this)
task = "regression"  # Example task

# Instantiate the BayesianCNN model with all required arguments
bayesian_cnn = bayesianCNN(num_classes, input_shape, kl_divergence_fn, task)

# Plot the model
plot_model(bayesian_cnn, to_file='model_plot.png', show_shapes=True)
