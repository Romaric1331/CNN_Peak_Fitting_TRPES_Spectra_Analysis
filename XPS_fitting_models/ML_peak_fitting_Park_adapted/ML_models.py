# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:54:06 2024

@author: ajulien & Romaric
"""

#%% Importation of packages
from keras.models import Model
from keras import layers
from keras.layers import (Input,Dense,BatchNormalization,Activation,Multiply,AveragePooling1D,Concatenate)
from keras.utils import plot_model
from tensorflow.keras.callbacks import (ModelCheckpoint,EarlyStopping,ReduceLROnPlateau)
import numpy as np
from math import sqrt
import tensorflow as tf

import tensorflow_probability as tfp
#%% Definition of Hyperparameter Bayes_sparse
# def scale_invariant_kl(q, p, _):
#     return tfp.distributions.kl_divergence(q, p) / tf.cast(tf.reduce_prod(q.batch_shape_tensor()), tf.float32)
# class BayesianConv1D(tf.keras.layers.Layer):
#     def __init__(self, filters, kernel_size, strides=1, padding="valid", activation=None, kernel_initializer='glorot_uniform',kl_weight=0.049082819079537994, **kwargs):
#         super(BayesianConv1D, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
#         self.activation = activation
#         self.kernel_initializer = kernel_initializer
#         self.conv = None

#     def build(self, input_shape):
#         self.conv = tfp.layers.Convolution1DFlipout(
#             filters=self.filters,
#             kernel_size=self.kernel_size,
#             strides=self.strides,
#             padding=self.padding,
#             activation=self.activation,
#             kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
#             kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(is_singular=False),
#             kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(tf.size(input=q.sample()), q.dtype),
#             bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
#             bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(is_singular=False),
#             bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(tf.size(input=q.sample()), q.dtype)
#         )
#         super(BayesianConv1D, self).build(input_shape)

#     def call(self, inputs):
#         return self.conv(inputs)

#     def get_config(self):
#         config = super(BayesianConv1D, self).get_config()
#         config.update({
#             "filters": self.filters,
#             "kernel_size": self.kernel_size,
#             "strides": self.strides,
#             "padding": self.padding,
#             "activation": self.activation,
#             "kernel_initializer": self.kernel_initializer
#         })
#         return config

# def define_H_bayesian_sparse_densenet(energy_n):
#     input_data = Input(shape=(energy_n, 1))
#     r = 16
#     Cf = 0.5
#     shortcut = 0
#     se=0
#     #cardinality = 16 what is this?
#     # resnet 1
#     x = BayesianConv1D(32, 4, strides=2, padding="same", kernel_initializer="he_normal")(input_data)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = BayesianConv1D(32, 4, strides=1, padding="same", kernel_initializer="he_normal")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = BayesianConv1D(32, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.MaxPooling1D(3, strides=2)(x)  # divide by 2
    
#     shortcut_dense = x
#     shortcut = x
#     shortcut = BayesianConv1D(64, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    
#     # ------------------ first layer-1
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     se = layers.GlobalAveragePooling1D()(x)  # global pooling
#     se = Dense(64 // r, kernel_initializer="he_normal")(se)  # FC
#     se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
#     se = Dense(64, kernel_initializer="he_normal")(se)  # FC
#     se = Activation("sigmoid")(se)  # Sigmoid
    
#     x = Multiply()([x, se])  # Scale
#     x = layers.Add()([x, shortcut])  # x.shape = (100,256)
    
    
#     # ----------------- first layer-2
#     shortcut = x
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(64 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(64, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
#     # se= Reshape([1,100])(se)
    
#     x = Multiply()([x, se])
#     x = layers.Add()([x, shortcut])
#     x = Concatenate()([x, shortcut_dense])  # Added layer into the concatenate
    
#     # se = layers.GlobalAveragePooling1D()(x)
#     # se = Dense(96 // r, kernel_initializer = 'he_normal')(se)
#     # se = layers.LeakyReLU(alpha = 0.01)(se)
#     # se = Dense(96, kernel_initializer = 'he_normal')(se)
#     # se = Activation('sigmoid')(se) # Sigmoid
    
#     # x = Multiply()([x,se])  # scale
    
    
#     # -----------------------------transition layer
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(int(96*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # 나누기 2
#     x = AveragePooling1D(3, padding="same", strides=2)(x)  # overlapped pooling
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(48 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(48, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
#     x = Multiply()([x, se])  # Scale
    
    
#     # --------------------------------------
#     # ----------------- second layer-1
#     shortcut_dense = x
    
#     shortcut = x
#     shortcut = BayesianConv1D(128, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     se = layers.GlobalAveragePooling1D()(x)  # global pooling
#     se = Dense(128 // r, kernel_initializer="he_normal")(se)  # FC
#     se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
#     se = Dense(128, kernel_initializer="he_normal")(se)  # FC
#     se = Activation("sigmoid")(se)  # Sigmoid
    
#     x = Multiply()([x, se])  # Scale
#     x = layers.Add()([x, shortcut])
    
    
#     # ----------------- second layer-2
#     shortcut = x
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(128 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(128, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
    
#     x = Multiply()([x, se])
#     x = layers.Add()([x, shortcut])
#     x = Concatenate()([x, shortcut_dense])
    
#     # se = layers.GlobalAveragePooling1D()(x)
#     # se = Dense(176 // r, kernel_initializer = 'he_normal')(se)
#     # se = layers.LeakyReLU(alpha = 0.01)(se)
#     # se = Dense(176, kernel_initializer = 'he_normal')(se)
#     # se = Activation('sigmoid')(se) # Sigmoid
    
#     # x = Multiply()([x,se])  # scale
    
    
#     # transition layer---------------------------------
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(int(176*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
#     x = AveragePooling1D(3, padding="same", strides=2)(x)
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(88 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(88, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
#     x = Multiply()([x, se])  # Scale
    
    
#     # --------------------------------------
#     # ----------------- third layer-1
#     shortcut_dense = x
    
#     shortcut = x
#     shortcut = BayesianConv1D(256, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     se = layers.GlobalAveragePooling1D()(x)  # global pooling
#     se = Dense(256 // r, kernel_initializer="he_normal")(se)  # FC
#     se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
#     se = Dense(256, kernel_initializer="he_normal")(se)  # FC
#     se = Activation("sigmoid")(se)  # Sigmoid
    
#     x = Multiply()([x, se])  # Scale
#     x = layers.Add()([x, shortcut])
    
    
#     # ----------------- third layer-2
#     shortcut = x
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(256 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(256, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
    
#     x = Multiply()([x, se])
#     x = layers.Add()([x, shortcut])
#     x = Concatenate()([x, shortcut_dense])
    
#     # se = layers.GlobalAveragePooling1D()(x)
#     # se = Dense(344 // r, kernel_initializer = 'he_normal')(se)
#     # se = layers.LeakyReLU(alpha = 0.01)(se)
#     # se = Dense(344, kernel_initializer = 'he_normal')(se)
#     # se = Activation('sigmoid')(se) # Sigmoid
    
#     # x = Multiply()([x,se])  # scale
    
    
#     # transition layer---------------------------------
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(int(334*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
#     x = AveragePooling1D(3, padding="same", strides=2)(x)
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(167 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(167, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
#     x = Multiply()([x, se])  # Scale
    
    
#     # --------------------------------------
#     # ----------------- four layer-1
#     shortcut_dense = x
#     shortcut = x
#     shortcut = BayesianConv1D(512, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     se = layers.GlobalAveragePooling1D()(x)  # global pooling
#     se = Dense(512 // r, kernel_initializer="he_normal")(se)  # FC
#     se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
#     se = Dense(512, kernel_initializer="he_normal")(se)  # FC
#     se = Activation("sigmoid")(se)  # Sigmoid
    
#     x = Multiply()([x, se])  # Scale
#     x = layers.Add()([x, shortcut])
    
    
#     # ----------------- four layer-2
#     shortcut = x
    
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     x = layers.BatchNormalization()(x)  # 786,994
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(512 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(512, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
    
#     x = Multiply()([x, se])
#     x = layers.Add()([x, shortcut])
#     x = Concatenate()([x, shortcut_dense])
    
#     # se = layers.GlobalAveragePooling1D()(x)
#     # se = Dense(679 // r, kernel_initializer = 'he_normal')(se)
#     # se = layers.LeakyReLU(alpha = 0.01)(se)
#     # se = Dense(679, kernel_initializer = 'he_normal')(se)
#     # se = Activation('sigmoid')(se) # Sigmoid
    
#     # x = Multiply()([x,se])  # scale
    
    
#     # transition layer---------------------------------
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = BayesianConv1D(int(679*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
#     x = AveragePooling1D(3, padding="same", strides=2)(x)
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(339 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(339, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
#     x = Multiply()([x, se])  # Scale
    
    
#     # --------------------------------------
#     # --------------------------------------
#     x = layers.GlobalAveragePooling1D()(x)
    
    
#     total_center1 = Dense(100, name="total_center1", kernel_initializer="he_normal")(x)
#     center_Batchnormalization = BatchNormalization()(total_center1)
#     total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
#     total_center3 = Dense(1, activation="linear", name="total_center3", kernel_initializer="he_normal")(total_center1_act)
    
#     total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(x)
#     width_Batchnormalization = BatchNormalization()(total_width1)
#     total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
#     total_width3 = Dense(1, activation="linear", name="total_width3", kernel_initializer="he_normal")(total_width1_act)
    
#     total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
#     amp_Batchnormalization = BatchNormalization()(total_amp1)
#     total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
#     total_amp3 = Dense(1, activation="linear", name="total_amp3", kernel_initializer="he_normal")(total_amp1_act)
    
#     total_peak_number1 = Dense(100, name="total_peak_number1", kernel_initializer="he_normal")(x)
#     peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
#     total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(peak_number_Batchnormalization)
#     total_peak_number3 = Dense(1, activation="linear", name="total_peak_number3", kernel_initializer="he_normal")(total_peak_number1_act)
    
    
#     model = Model(inputs=input_data,outputs=[total_center3, total_width3, total_amp3, total_peak_number3])
    
#     return model

#%% Definition of Bayesian_sparse_densenet architetcure

def scale_invariant_kl(q, p, _):
    return tfp.distributions.kl_divergence(q, p) / tf.cast(tf.reduce_prod(q.batch_shape_tensor()), tf.float32)
class BayesianConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="valid", activation=None, kernel_initializer='glorot_uniform', **kwargs):
        super(BayesianConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.conv = None

    def build(self, input_shape):
        self.conv = tfp.layers.Convolution1DFlipout(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(tf.size(input=q.sample()), q.dtype),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(tf.size(input=q.sample()), q.dtype)
        )
        super(BayesianConv1D, self).build(input_shape)

    def call(self, inputs):
        return self.conv(inputs)

    def get_config(self):
        config = super(BayesianConv1D, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer
        })
        return config
def define_bayesian_sparse_densenet(energy_n):
    input_data = Input(shape=(energy_n, 1))
    r = 16
    Cf = 0.5
    shortcut = 0
    se=0
    #cardinality = 16 what is this?
    # resnet 1
    x = BayesianConv1D(32, 4, strides=2, padding="same", kernel_initializer="he_normal")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    x = BayesianConv1D(32, 4, strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    x = BayesianConv1D(32, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    x = layers.MaxPooling1D(3, strides=2)(x)  # divide by 2
    
    shortcut_dense = x
    shortcut = x
    shortcut = BayesianConv1D(64, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    
    # ------------------ first layer-1
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(64 // r, kernel_initializer="he_normal")(se)  # FC
    se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(64, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x = layers.Add()([x, shortcut])  # x.shape = (100,256)
    
    
    # ----------------- first layer-2
    shortcut = x
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(64 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(64, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    # se= Reshape([1,100])(se)
    
    x = Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])  # Added layer into the concatenate
    
    # se = layers.GlobalAveragePooling1D()(x)
    # se = Dense(96 // r, kernel_initializer = 'he_normal')(se)
    # se = layers.LeakyReLU(alpha = 0.01)(se)
    # se = Dense(96, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # -----------------------------transition layer
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(int(96*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # 나누기 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)  # overlapped pooling
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(48 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(48, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # ----------------- second layer-1
    shortcut_dense = x
    
    shortcut = x
    shortcut = BayesianConv1D(128, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(128 // r, kernel_initializer="he_normal")(se)  # FC
    se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(128, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x = layers.Add()([x, shortcut])
    
    
    # ----------------- second layer-2
    shortcut = x
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(128 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(128, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])
    
    # se = layers.GlobalAveragePooling1D()(x)
    # se = Dense(176 // r, kernel_initializer = 'he_normal')(se)
    # se = layers.LeakyReLU(alpha = 0.01)(se)
    # se = Dense(176, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # transition layer---------------------------------
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(int(176*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(88 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(88, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # ----------------- third layer-1
    shortcut_dense = x
    
    shortcut = x
    shortcut = BayesianConv1D(256, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(256 // r, kernel_initializer="he_normal")(se)  # FC
    se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(256, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x = layers.Add()([x, shortcut])
    
    
    # ----------------- third layer-2
    shortcut = x
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(256 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(256, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])
    
    # se = layers.GlobalAveragePooling1D()(x)
    # se = Dense(344 // r, kernel_initializer = 'he_normal')(se)
    # se = layers.LeakyReLU(alpha = 0.01)(se)
    # se = Dense(344, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # transition layer---------------------------------
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(int(334*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(167 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(167, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # ----------------- four layer-1
    shortcut_dense = x
    shortcut = x
    shortcut = BayesianConv1D(512, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(512 // r, kernel_initializer="he_normal")(se)  # FC
    se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(512, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x = layers.Add()([x, shortcut])
    
    
    # ----------------- four layer-2
    shortcut = x
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)  # 786,994
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(512 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(512, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])
    
    # se = layers.GlobalAveragePooling1D()(x)
    # se = Dense(679 // r, kernel_initializer = 'he_normal')(se)
    # se = layers.LeakyReLU(alpha = 0.01)(se)
    # se = Dense(679, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # transition layer---------------------------------
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(int(679*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(339 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(339, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # --------------------------------------
    x = layers.GlobalAveragePooling1D()(x)
    
    
    total_center1 = Dense(100, name="total_center1", kernel_initializer="he_normal")(x)
    center_Batchnormalization = BatchNormalization()(total_center1)
    total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
    total_center3 = Dense(1, activation="linear", name="total_center3", kernel_initializer="he_normal")(total_center1_act)
    
    total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(x)
    width_Batchnormalization = BatchNormalization()(total_width1)
    total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
    total_width3 = Dense(1, activation="linear", name="total_width3", kernel_initializer="he_normal")(total_width1_act)
    
    total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
    amp_Batchnormalization = BatchNormalization()(total_amp1)
    total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
    total_amp3 = Dense(1, activation="linear", name="total_amp3", kernel_initializer="he_normal")(total_amp1_act)
    
    total_peak_number1 = Dense(100, name="total_peak_number1", kernel_initializer="he_normal")(x)
    peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
    total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(peak_number_Batchnormalization)
    total_peak_number3 = Dense(1, activation="linear", name="total_peak_number3", kernel_initializer="he_normal")(total_peak_number1_act)
    
    
    model = Model(inputs=input_data,outputs=[total_center3, total_width3, total_amp3, total_peak_number3])
    
    return model


#%% Definition of BayesianCNN architetcure
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, Model

def scale_invariant_kl(q, p, _):
    return tfp.distributions.kl_divergence(q, p) / tf.cast(tf.reduce_prod(q.batch_shape_tensor()), tf.float32)

class MyConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation, name=None, trainable=True, **kwargs):
        super(MyConvLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.conv_layer = None

    def build(self, input_shape):
        self.conv_layer = tfp.layers.Convolution1DFlipout(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_divergence_fn=scale_invariant_kl,
            bias_divergence_fn=scale_invariant_kl,
            activation=self.activation,
            trainable=self.trainable
        )
        super(MyConvLayer, self).build(input_shape)

    def call(self, inputs):
        return self.conv_layer(inputs)

    def get_config(self):
        config = super(MyConvLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def define_bayesian_cnn(energy_n):
    strides = 1
    pool_size = 2
    average_pool_layer = layers.AveragePooling1D(pool_size=pool_size)
    output_act = None  # No activation for regression tasks
    prob_act = "softplus"

    # Input Layer
    input_data = layers.Input(shape=(energy_n, 1), name="input_1")

    # Convolutional Layers
    conv_1_short = MyConvLayer(filters=32, kernel_size=4, strides=strides, padding="same", activation=prob_act, name="conv_1_short")(input_data)
    conv_1_medium = MyConvLayer(filters=32, kernel_size=4, strides=strides, padding="same", activation=prob_act, name="conv_1_medium")(input_data)
    conv_1_long = MyConvLayer(filters=32, kernel_size=3, strides=strides, padding="same", activation=prob_act, name="conv_1_long")(input_data)

    merged_sublayers = layers.concatenate([conv_1_short, conv_1_medium, conv_1_long])

    conv_2 = MyConvLayer(filters=64, kernel_size=3, strides=strides, padding="same", activation=prob_act, name="conv_2")(merged_sublayers)
    conv_3 = MyConvLayer(filters=128, kernel_size=3, strides=strides, padding="same", activation=prob_act, name="conv_3")(conv_2)
    
    average_pool_1 = average_pool_layer(conv_3)
    flatten_1 = layers.Flatten(name="flatten1")(average_pool_1)
    drop_1 = layers.Dropout(rate=0.2, name="drop_1")(flatten_1)

    # Set 1 Outputs
    dense2_1_set1 = layers.Dense(512, name="dense2_1_set1")(drop_1)
    batch_norm_1_set1 = layers.BatchNormalization(name="batch_norm_1_set1")(dense2_1_set1)
    leaky_relu_1_set1 = layers.LeakyReLU(name="leaky_relu_1_set1")(batch_norm_1_set1)
    center_set1 = layers.Dense(1, activation=output_act, name="center_set1")(leaky_relu_1_set1)
    width_set1 = layers.Dense(1, activation=output_act, name="width_set1")(leaky_relu_1_set1)
    amplitude_set1 = layers.Dense(1, activation=output_act, name="amplitude_set1")(leaky_relu_1_set1)
    peak_number_set1 = layers.Dense(1, activation=output_act, name="peak_number_set1")(leaky_relu_1_set1)

    # Set 2 Outputs
    dense2_1_set2 = layers.Dense(512, name="dense2_1_set2")(drop_1)
    batch_norm_1_set2 = layers.BatchNormalization(name="batch_norm_1_set2")(dense2_1_set2)
    leaky_relu_1_set2 = layers.LeakyReLU(name="leaky_relu_1_set2")(batch_norm_1_set2)
    center_set2 = layers.Dense(1, activation=output_act, name="center_set2")(leaky_relu_1_set2)
    width_set2 = layers.Dense(1, activation=output_act, name="width_set2")(leaky_relu_1_set2)
    amplitude_set2 = layers.Dense(1, activation=output_act, name="amplitude_set2")(leaky_relu_1_set2)
    peak_number_set2 = layers.Dense(1, activation=output_act, name="peak_number_set2")(leaky_relu_1_set2)

    # Set 3 Outputs
    dense2_1_set3 = layers.Dense(512, name="dense2_1_set3")(drop_1)
    batch_norm_1_set3 = layers.BatchNormalization(name="batch_norm_1_set3")(dense2_1_set3)
    leaky_relu_1_set3 = layers.LeakyReLU(name="leaky_relu_1_set3")(batch_norm_1_set3)
    center_set3 = layers.Dense(1, activation=output_act, name="center_set3")(leaky_relu_1_set3)
    width_set3 = layers.Dense(1, activation=output_act, name="width_set3")(leaky_relu_1_set3)
    amplitude_set3 = layers.Dense(1, activation=output_act, name="amplitude_set3")(leaky_relu_1_set3)
    peak_number_set3 = layers.Dense(1, activation=output_act, name="peak_number_set3")(leaky_relu_1_set3)

    # Set 4 Outputs
    dense2_1_set4 = layers.Dense(512, name="dense2_1_set4")(drop_1)
    batch_norm_1_set4 = layers.BatchNormalization(name="batch_norm_1_set4")(dense2_1_set4)
    leaky_relu_1_set4 = layers.LeakyReLU(name="leaky_relu_1_set4")(batch_norm_1_set4)
    center_set4 = layers.Dense(1, activation=output_act, name="center_set4")(leaky_relu_1_set4)
    width_set4 = layers.Dense(1, activation=output_act, name="width_set4")(leaky_relu_1_set4)
    amplitude_set4 = layers.Dense(1, activation=output_act, name="amplitude_set4")(leaky_relu_1_set4)
    peak_number_set4 = layers.Dense(1, activation=output_act, name="peak_number_set4")(leaky_relu_1_set4)

    # Define the model with all outputs
    outputs = [
        center_set1, width_set1, amplitude_set1, peak_number_set1,
        center_set2, width_set2, amplitude_set2, peak_number_set2,
        center_set3, width_set3, amplitude_set3, peak_number_set3,
        center_set4, width_set4, amplitude_set4, peak_number_set4
    ]

    model = Model(inputs=input_data, outputs=outputs, name="BayesianCNN")

    return model

#%% Definition of sparse_densenet architetcure
def define_sparse_densenet(energy_n):
    # energy_n: int, size of the spectrum
    
    # ### SE- Dense- Resnet
    #
    # #### Densnet
    # - concept : channel의 reuse
    # - How
    # - -  i) transition layer 사용하여 parameter 경량화 (=composition factor 0.5=논문에서 추천한 값)
    # - - ii) concatenate로  projection block 을 connecting
    # - review
    # - - i) 모든 residual block에 connecting 하는 것보다  projection conection쓰이는 곳에만 연결하는게 더 좋은 효과
    #
    # #### The translation of the Korean text into English is:
    #
    # - Concept: Reuse of channels in channel-wise attention.
    # - How:
    #   - i) Use transition layer to reduce the number of parameters (= recommended value of composition factor 0.5 in the paper).
    #   - ii) Connect projection block with concatenate.
    # - Review:
    #   - i) Connecting only where the projection connection is used is more effective than connecting to all residual blocks.
    #
    #
    
    # Sparse dense block with no se block
    input_data = Input(shape=(energy_n, 1))
    r = 16
    shortcut = 0
    se = 0
    Cf = 0.5
    cardinality = 16
    
    # resnet 1
    x = layers.Conv1D(32, 4, strides=2, padding="same", kernel_initializer="he_normal")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    x = layers.Conv1D(32, 4, strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    x = layers.Conv1D(32, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    
    x = layers.MaxPooling1D(3, strides=2)(x)  # divide by 2
    
    shortcut_dense = x
    shortcut = x
    shortcut = layers.Conv1D(64, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    
    # ------------------ first layer-1
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(64 // r, kernel_initializer="he_normal")(se)  # FC
    se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(64, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x = layers.Add()([x, shortcut])  # x.shape = (100,256)
    
    
    # ----------------- first layer-2
    shortcut = x
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(64 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(64, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    # se= Reshape([1,100])(se)
    
    x = Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])  # Added layer into the concatenate
    
    # se = layers.GlobalAveragePooling1D()(x)
    # se = Dense(96 // r, kernel_initializer = 'he_normal')(se)
    # se = layers.LeakyReLU(alpha = 0.01)(se)
    # se = Dense(96, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # -----------------------------transition layer
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(int(96*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # 나누기 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)  # overlapped pooling
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(48 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(48, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # ----------------- second layer-1
    shortcut_dense = x
    
    shortcut = x
    shortcut = layers.Conv1D(128, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(128 // r, kernel_initializer="he_normal")(se)  # FC
    se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(128, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x = layers.Add()([x, shortcut])
    
    
    # ----------------- second layer-2
    shortcut = x
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(128 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(128, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])
    
    # se = layers.GlobalAveragePooling1D()(x)
    # se = Dense(176 // r, kernel_initializer = 'he_normal')(se)
    # se = layers.LeakyReLU(alpha = 0.01)(se)
    # se = Dense(176, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # transition layer---------------------------------
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(int(176*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(88 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(88, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # ----------------- third layer-1
    shortcut_dense = x
    
    shortcut = x
    shortcut = layers.Conv1D(256, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(256 // r, kernel_initializer="he_normal")(se)  # FC
    se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(256, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x = layers.Add()([x, shortcut])
    
    
    # ----------------- third layer-2
    shortcut = x
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(256 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(256, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])
    
    # se = layers.GlobalAveragePooling1D()(x)
    # se = Dense(344 // r, kernel_initializer = 'he_normal')(se)
    # se = layers.LeakyReLU(alpha = 0.01)(se)
    # se = Dense(344, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # transition layer---------------------------------
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(int(334*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(167 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(167, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # ----------------- four layer-1
    shortcut_dense = x
    shortcut = x
    shortcut = layers.Conv1D(512, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(512 // r, kernel_initializer="he_normal")(se)  # FC
    se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(512, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x = layers.Add()([x, shortcut])
    
    
    # ----------------- four layer-2
    shortcut = x
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = layers.BatchNormalization()(x)  # 786,994
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(512 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(512, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])
    
    # se = layers.GlobalAveragePooling1D()(x)
    # se = Dense(679 // r, kernel_initializer = 'he_normal')(se)
    # se = layers.LeakyReLU(alpha = 0.01)(se)
    # se = Dense(679, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # transition layer---------------------------------
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.Conv1D(int(679*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)
    
    se = layers.GlobalAveragePooling1D()(x)
    se = Dense(339 // r, kernel_initializer="he_normal")(se)
    se = layers.LeakyReLU(alpha=0.01)(se)
    se = Dense(339, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # --------------------------------------
    x = layers.GlobalAveragePooling1D()(x)
    
    
    total_center1 = Dense(100, name="total_center1", kernel_initializer="he_normal")(x)
    center_Batchnormalization = BatchNormalization()(total_center1)
    total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
    total_center3 = Dense(1, activation="linear", name="total_center3", kernel_initializer="he_normal")(total_center1_act)
    
    total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(x)
    width_Batchnormalization = BatchNormalization()(total_width1)
    total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
    total_width3 = Dense(1, activation="linear", name="total_width3", kernel_initializer="he_normal")(total_width1_act)
    
    total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
    amp_Batchnormalization = BatchNormalization()(total_amp1)
    total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
    total_amp3 = Dense(1, activation="linear", name="total_amp3", kernel_initializer="he_normal")(total_amp1_act)
    
    total_peak_number1 = Dense(100, name="total_peak_number1", kernel_initializer="he_normal")(x)
    peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
    total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(peak_number_Batchnormalization)
    total_peak_number3 = Dense(1, activation="linear", name="total_peak_number3", kernel_initializer="he_normal")(total_peak_number1_act)
    
    
    model = Model(inputs=input_data,outputs=[total_center3, total_width3, total_amp3, total_peak_number3])
    
    return model
#%% Definition of SEResNet

# #%% Definition of SEResNet architecture
# def define_SEResNet(energy_n):
#     # energy_n: int, size of the spectrum
    
#     # ### SE-Resnet
#     #
#     # #### SE-block
#     # - concept
#     # - - i) 압축, 펌핑을 통한 channel의 재보정
#     # - - ii) 10%이내의 적은 parameter투자를 통해 성능향상
#     # - - iii) flexible 하여 모든 model에 연결 가능
#     #
#     # - How
#     # - - i) GlobalAveragePooling1D를 통해 압축
#     # - - ii) r (= Reduction ratio) = 16 (논문에서 추천한 값) 값을 이용하여 점차적으로 펌핑
#     # - - iii) 모든 residual block에 사용 (identity connection, projection connection)
#     #
    
#     # origin Se-resnet
#     input_data = Input(shape=(energy_n, 1))
#     r = 16
#     shortcut = 0
#     se = 0
    
#     # resnet 1차
#     x = layers.Conv1D(
#         32, 4, strides=2, padding="same", kernel_initializer="he_normal"
#     )(input_data)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = layers.Conv1D(
#         32, 4, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = layers.Conv1D(
#         32, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = layers.MaxPooling1D(3, strides=2)(x)  # 나누기 2
#     # 443
#     # --------------------------------------
    
#     shortcut = x
#     shortcut = layers.Conv1D(
#         64, 1, strides=1, padding="valid", kernel_initializer="he_normal"
#     )(shortcut)
    
    
#     x = layers.Conv1D(
#         64, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         64, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     se = layers.GlobalAveragePooling1D()(x)  # global pooling
#     se = Dense(64 // r, kernel_initializer="he_normal")(se)  # FC
#     se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
#     se = Dense(64, kernel_initializer="he_normal")(se)  # FC
#     se = Activation("sigmoid")(se)  # Sigmoid
#     # se= Reshape([1,64])(se)
    
#     x = Multiply()([x, se])  # Scale
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
    
#     shortcut = x
#     x = layers.Conv1D(
#         64, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         64, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(64 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(64, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
#     # se= Reshape([1,64])(se)
    
#     x = Multiply()([x, se])
    
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
    
#     # --------------------------------------
    
#     shortcut = x
#     shortcut = layers.Conv1D(
#         128, 1, strides=2, padding="valid", kernel_initializer="he_normal"
#     )(shortcut)
    
#     x = layers.Conv1D(
#         128, 3, strides=2, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # 나누기 2
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         128, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     se = layers.GlobalAveragePooling1D()(x)  # global pooling
#     se = Dense(128 // r, kernel_initializer="he_normal")(se)  # FC
#     se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
#     se = Dense(128, kernel_initializer="he_normal")(se)  # FC
#     se = Activation("sigmoid")(se)  # Sigmoid
#     # se= Reshape([1,128])(se)
    
#     x = Multiply()([x, se])  # Scale
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     shortcut = x
#     x = layers.Conv1D(
#         128, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # identity shortcut
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         128, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(128 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(128, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
#     # se= Reshape([1,128])(se)
    
#     x = Multiply()([x, se])
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
    
#     # --------------------------------------
    
#     shortcut = x
#     shortcut = layers.Conv1D(
#         256, 1, strides=2, padding="valid", kernel_initializer="he_normal"
#     )(shortcut)
    
#     x = layers.Conv1D(
#         256, 3, strides=2, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # 나누기 2
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         256, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     se = layers.GlobalAveragePooling1D()(x)  # global pooling
#     se = Dense(256 // r, kernel_initializer="he_normal")(se)  # FC
#     se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
#     se = Dense(256, kernel_initializer="he_normal")(se)  # FC
#     se = Activation("sigmoid")(se)  # Sigmoid
#     # se= Reshape([1,256])(se)
    
#     x = Multiply()([x, se])  # Scale
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     shortcut = x
#     x = layers.Conv1D(
#         256, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # identity shortcut
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         256, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(256 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(256, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
#     # se= Reshape([1,256])(se)
    
#     x = Multiply()([x, se])
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     # --------------------------------------
    
#     shortcut = x
#     shortcut = layers.Conv1D(
#         512, 1, strides=2, padding="valid", kernel_initializer="he_normal"
#     )(shortcut)
    
#     x = layers.Conv1D(
#         512, 3, strides=2, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # 나누기 2
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         512, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     se = layers.GlobalAveragePooling1D()(x)  # global pooling
#     se = Dense(512 // r, kernel_initializer="he_normal")(se)  # FC
#     se = layers.LeakyReLU(alpha=0.01)(se)  # ReLU
#     se = Dense(512, kernel_initializer="he_normal")(se)  # FC
#     se = Activation("sigmoid")(se)  # Sigmoid
#     # se= Reshape([1,512])(se)
    
#     x = Multiply()([x, se])  # Scale
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     shortcut = x
#     x = layers.Conv1D(
#         512, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # identity shortcut
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         512, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     se = layers.GlobalAveragePooling1D()(x)
#     se = Dense(512 // r, kernel_initializer="he_normal")(se)
#     se = layers.LeakyReLU(alpha=0.01)(se)
#     se = Dense(512, kernel_initializer="he_normal")(se)
#     se = Activation("sigmoid")(se)  # Sigmoid
#     # se= Reshape([1,512])(se)
    
#     x = Multiply()([x, se])
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
    
#     # --------------------------------------
    
#     x = layers.GlobalAveragePooling1D()(x)
    
#     # and BN을 확인해보자
    
    
#     total_center1 = Dense(
#         100, name="total_center1", kernel_initializer="he_normal"
#     )(x)
#     center_Batchnormalization = BatchNormalization()(total_center1)
#     total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
#     total_center3 = Dense(
#         1,
#         activation="linear",
#         name="total_center3",
#         kernel_initializer="he_normal",
#     )(total_center1_act)
    
#     total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
#         x
#     )
#     width_Batchnormalization = BatchNormalization()(total_width1)
#     total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
#     total_width3 = Dense(
#         1, activation="linear", name="total_width3", kernel_initializer="he_normal"
#     )(total_width1_act)
    
#     total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
#     amp_Batchnormalization = BatchNormalization()(total_amp1)
#     total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
#     total_amp3 = Dense(
#         1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
#     )(total_amp1_act)
    
#     total_peak_number1 = Dense(
#         100, name="total_peak_number1", kernel_initializer="he_normal"
#     )(x)
#     peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
#     total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
#         peak_number_Batchnormalization
#     )
#     total_peak_number3 = Dense(
#         1,
#         activation="linear",
#         name="total_peak_number3",
#         kernel_initializer="he_normal",
#     )(total_peak_number1_act)
    
    
#     model = Model(inputs=input_data,outputs=[total_center3, total_width3, total_amp3, total_peak_number3])
    
#     return model


#%% Definition of ResNet architetcure
# def define_ResNet(energy_n):
#     # energy_n: int, size of the spectrum
    
#     # ### Resnet
#     # - concept
#     # - -  i) residual connection을 통해 function 재설정
#     # - How
#     # - - i) pre activation( 논문에서 추천한 sumsampling순서)
#     # - - ii) 쌓을수록 resnet의 장점이 두드러지지만 vggnet과 비교를 위해 4개의 block으로 8개의 layer을 쌓음
#     # - review
#     # - - (64x2-128x2-256x2-512x2 총 8개의 convolution layers 사용)
#     # - - resnet의 장점인 depth의 극대화를 하지 않고 vggnet과 같이 8layer밖에 되지 않아 성능이 조금밖에 차이가 없음
#     #
#     #
#     # The translation of the Korean text into English is:
#     # ##### Resnet
#     #
#     # - Concept:
#     #     -- i) Resetting function through residual connections.
#     # - How:
#     #     -- i) Pre-activation (recommended subsampling order in the paper).
#     #     -- ii) Stacked with 4 blocks and 8 layers to compare with VGGNet and highlight the advantages of ResNet.
#     # - Review:
#     #     Used a total of 8 convolution layers (64x2-128x2-256x2-512x2).
#     #     Performance difference is not significant compared to VGGNet, as ResNet does not maximize the depth like VGGNet and is limited to only 8 layers.
    
#     input_data = Input(shape=(energy_n, 1))
#     r = 16
#     # /gpu:0
#     # resnet 1차
#     x = layers.Conv1D(
#         32, 4, strides=2, padding="same", kernel_initializer="he_normal"
#     )(input_data)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = layers.Conv1D(
#         32, 4, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = layers.Conv1D(
#         32, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
#     x = layers.MaxPooling1D(3, strides=2)(x)  # 나누기 2
    
#     # --------------------------------------
    
#     shortcut = x
#     shortcut = layers.Conv1D(
#         64, 1, strides=1, padding="valid", kernel_initializer="he_normal"
#     )(shortcut)
    
    
#     x = layers.Conv1D(
#         64, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         64, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
    
#     shortcut = x  # identity shortcut
#     x = layers.Conv1D(
#         64, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         64, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
    
#     # --------------------------------------
    
#     shortcut = x
#     shortcut = layers.Conv1D(
#         128, 1, strides=2, padding="valid", kernel_initializer="he_normal"
#     )(shortcut)
    
#     x = layers.Conv1D(
#         128, 3, strides=2, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # 나누기 2
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         128, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
    
#     shortcut = x  # identity shortcut
#     x = layers.Conv1D(
#         128, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # identity shortcut
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         128, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     # --------------------------------------
    
#     shortcut = x
#     shortcut = layers.Conv1D(
#         256, 1, strides=2, padding="valid", kernel_initializer="he_normal"
#     )(shortcut)
    
#     x = layers.Conv1D(
#         256, 3, strides=2, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # 나누기 2
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         256, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
    
#     shortcut = x  # identity shortcut
#     x = layers.Conv1D(
#         256, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # identity shortcut
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         256, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     # --------------------------------------
    
#     shortcut = x
#     shortcut = layers.Conv1D(
#         512, 1, strides=2, padding="valid", kernel_initializer="he_normal"
#     )(shortcut)
    
#     x = layers.Conv1D(
#         512, 3, strides=2, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # 나누기 2
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         512, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
    
#     shortcut = x  # identity shortcut
#     x = layers.Conv1D(
#         512, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(
#         x
#     )  # identity shortcut
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     x = layers.Conv1D(
#         512, 3, strides=1, padding="same", kernel_initializer="he_normal"
#     )(x)
#     x = layers.BatchNormalization()(x)
    
    
#     x = layers.Add()([x, shortcut])
#     x = layers.LeakyReLU(alpha=0.01)(x)
    
#     # --------------------------------------
    
#     x = layers.GlobalAveragePooling1D()(x)
    
#     # and BN을 확인해보자
    
    
#     total_center1 = Dense(
#         100, name="total_center1", kernel_initializer="he_normal"
#     )(x)
#     center_Batchnormalization = BatchNormalization()(total_center1)
#     total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
#     total_center3 = Dense(
#         1,
#         activation="linear",
#         name="total_center3",
#         kernel_initializer="he_normal",
#     )(total_center1_act)
    
#     total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
#         x
#     )
#     width_Batchnormalization = BatchNormalization()(total_width1)
#     total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
#     total_width3 = Dense(
#         1, activation="linear", name="total_width3", kernel_initializer="he_normal"
#     )(total_width1_act)
    
#     total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
#     amp_Batchnormalization = BatchNormalization()(total_amp1)
#     total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
#     total_amp3 = Dense(
#         1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
#     )(total_amp1_act)
    
#     total_peak_number1 = Dense(
#         100, name="total_peak_number1", kernel_initializer="he_normal"
#     )(x)
#     peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
#     total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
#         peak_number_Batchnormalization
#     )
#     total_peak_number3 = Dense(
#         1,
#         activation="linear",
#         name="total_peak_number3",
#         kernel_initializer="he_normal",
#     )(total_peak_number1_act)
    
#     model = Model(inputs=input_data,outputs=[total_center3, total_width3, total_amp3, total_peak_number3])
    
#     return model


#%% Definition of VGGNet architetcure
# def define_VGGNet(energy_n):
#     # energy_n: int, size of the spectrum
    
#     # ### Vggnet
#     # - concent : 인수분해된 filter size로 반복 극대화
#     # - how :
#     # - -  i) filter fize=4,3의 convolution을 3,2개를 한꺼번에 쌓은후 subsampling
#     # - - ii) 32x3-64x2-128x2-256x2-512x2
    
#     input_data = Input(shape=(energy_n, 1))
    
#     x = layers.Conv1D(32, 4, strides=2, activation="relu", padding="same")(
#         input_data
#     )
#     x = layers.Conv1D(32, 4, strides=1, activation="relu", padding="same")(x)
#     x = layers.Conv1D(32, 3, strides=1, activation="relu", padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling1D(2, strides=2)(x)
    
#     x = layers.Conv1D(64, 3, strides=1, activation="relu", padding="same")(x)
#     x = layers.Conv1D(64, 3, strides=1, activation="relu", padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling1D(2, strides=2)(x)
    
#     x = layers.Conv1D(128, 3, strides=1, activation="relu", padding="same")(x)
#     x = layers.Conv1D(128, 3, strides=1, activation="relu", padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling1D(2, strides=2)(x)
    
#     x = layers.Conv1D(256, 3, strides=1, activation="relu", padding="same")(x)
#     x = layers.Conv1D(256, 3, strides=1, activation="relu", padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling1D(2, strides=2)(x)
    
#     x = layers.Conv1D(512, 3, strides=1, activation="relu", padding="same")(x)
#     x = layers.Conv1D(512, 3, strides=1, activation="relu", padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling1D(2, strides=2)(x)
    
#     x = layers.GlobalMaxPooling1D()(x)
    
    
#     total_center1 = Dense(
#         100, name="total_center1", kernel_initializer="he_normal"
#     )(x)
#     center_Batchnormalization = BatchNormalization()(total_center1)
#     total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
#     total_center3 = Dense(
#         1,
#         activation="linear",
#         name="total_center3",
#         kernel_initializer="he_normal",
#     )(total_center1_act)
    
#     total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
#         x
#     )
#     width_Batchnormalization = BatchNormalization()(total_width1)
#     total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
#     total_width3 = Dense(
#         1, activation="linear", name="total_width3", kernel_initializer="he_normal"
#     )(total_width1_act)
    
#     total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
#     amp_Batchnormalization = BatchNormalization()(total_amp1)
#     total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
#     total_amp3 = Dense(
#         1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
#     )(total_amp1_act)
    
#     total_peak_number1 = Dense(
#         100, name="total_peak_number1", kernel_initializer="he_normal"
#     )(x)
#     peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
#     total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
#         peak_number_Batchnormalization
#     )
#     total_peak_number3 = Dense(
#         1,
#         activation="linear",
#         name="total_peak_number3",
#         kernel_initializer="he_normal",
#     )(total_peak_number1_act)
    
#     model = Model(inputs=input_data,outputs=[total_center3, total_width3, total_amp3, total_peak_number3])

#     return model


#%% Definition of AlexNet architetcure
# def define_Alex_ZFNet(energy_n):
#     # energy_n: int, size of the spectrum
    
#     # Alexnet #############################################################
#     # -concent :
#     # - - i) input data에 맞는 다양한 filter size 사용
#     # - - ii) conv-pooling의 단순한 subsampling 반복 탈피
#     # - - iii) overlapped pooling
#     # - How
#     # - - i) 중간 conv1d 2개의 layer는subsampling 안함
#     # - - ii) 96x1 -256x1 - 384x3 의 channel
    
#     # alexnet+zfnet
#     input_data = Input(shape=(energy_n, 1))
    
#     x = layers.Conv1D(96, 20, strides=2, activation="relu", padding="same")(
#         input_data
#     )
#     x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
#     x = layers.Conv1D(256, 9, strides=2, activation="relu", padding="same")(x)
#     x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
#     x = layers.Conv1D(384, 4, activation="relu", padding="same")(x)
#     x = layers.Conv1D(384, 4, activation="relu", padding="same")(x)
#     x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
#     x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
#     x = layers.GlobalMaxPooling1D()(x)
    
    
#     total_center1 = Dense(
#         100, name="total_center1", kernel_initializer="he_normal"
#     )(x)
#     center_Batchnormalization = BatchNormalization()(total_center1)
#     total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
#     total_center3 = Dense(
#         1,
#         activation="linear",
#         name="total_center3",
#         kernel_initializer="he_normal",
#     )(total_center1_act)
    
#     total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
#         x
#     )
#     width_Batchnormalization = BatchNormalization()(total_width1)
#     total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
#     total_width3 = Dense(
#         1, activation="linear", name="total_width3", kernel_initializer="he_normal"
#     )(total_width1_act)
    
#     total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
#     amp_Batchnormalization = BatchNormalization()(total_amp1)
#     total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
#     total_amp3 = Dense(
#         1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
#     )(total_amp1_act)
    
#     total_peak_number1 = Dense(
#         200, name="total_peak_number1", kernel_initializer="he_normal"
#     )(x)
#     peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
#     total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
#         peak_number_Batchnormalization
#     )
#     total_peak_number3 = Dense(
#         1,
#         activation="linear",
#         name="total_peak_number3",
#         kernel_initializer="he_normal",
#     )(total_peak_number1_act)
    
    
#     model = Model(inputs=input_data,outputs=[total_center3, total_width3, total_amp3, total_peak_number3])
    
#     return model


#%% Definition of LeNet architetcure - Tried out Bayesian parameters, if needed it can be changed
def define_LeNet(energy_n):
    # energy_n: int, size of the spectrum
    
    # ### Lenet
    # - concept:
    # - - i) convolution layer의 첫 사용
    # - - ii) 단순한 conv-pooling의 반복단계
    # - - iii) 이전 peak fitting 논문의 cnn model
    # - How
    # - - i) conv-subsampling을 한개의 block으로 총  4 개의 block 쌓음
    # - - ii) channel 32x1-64x1-128x1-256x1
    
    input_data = Input(shape=(energy_n, 1))
    
    x = BayesianConv1D(32, 100, strides=3, activation="relu")(input_data)
    x = layers.MaxPooling1D(2)(x)
    
    x = BayesianConv1D(64, 10, strides=2, activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = BayesianConv1D(128, 4, activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = BayesianConv1D(256, 2, activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.GlobalMaxPooling1D()(x)
    
    
    total_center1 = Dense(
        100, name="total_center1", kernel_initializer="he_normal"
    )(x)
    center_Batchnormalization = BatchNormalization()(total_center1)
    total_center1_act = layers.LeakyReLU(alpha=0.01)(center_Batchnormalization)
    total_center3 = Dense(
        1,
        activation="linear",
        name="total_center3",
        kernel_initializer="he_normal",
    )(total_center1_act)
    
    total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(
        x
    )
    width_Batchnormalization = BatchNormalization()(total_width1)
    total_width1_act = layers.LeakyReLU(alpha=0.01)(width_Batchnormalization)
    total_width3 = Dense(
        1, activation="linear", name="total_width3", kernel_initializer="he_normal"
    )(total_width1_act)
    
    total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
    amp_Batchnormalization = BatchNormalization()(total_amp1)
    total_amp1_act = layers.LeakyReLU(alpha=0.01)(amp_Batchnormalization)
    total_amp3 = Dense(
        1, activation="linear", name="total_amp3", kernel_initializer="he_normal"
    )(total_amp1_act)
    
    
    total_peak_number1 = Dense(
        100, name="total_peak_number1", kernel_initializer="he_normal"
    )(x)
    peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
    total_peak_number1_act = layers.LeakyReLU(alpha=0.01)(
        peak_number_Batchnormalization
    )
    total_peak_number3 = Dense(
        1,
        activation="linear",
        name="total_peak_number3",
        kernel_initializer="he_normal",
    )(total_peak_number1_act)
    
    model = Model(inputs=input_data,outputs=[total_center3, total_width3, total_amp3, total_peak_number3])
    
    return model


#%% Print summary of model in .txt file
def print_summary(model,model_summary_file):
    # model: keras model
    # model_summary_file: str for the .txt file to store the model summary
    
    # Get summary in a str
    stringlist = []
    model.summary(print_fn=lambda x,line_break: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    
    # Write in .txt file
    with open(model_summary_file, 'w', encoding="utf-8") as f:
        f.write(short_model_summary)
    
    return


#%% Plot summary of model in .png image
def plot_summary(model,model_summary_file):
    plot_model(model,to_file=model_summary_file,show_shapes=True)


#%% Compile, setup, train and validate model
# def compile_setup_train_validate(model,training_data,validation_data,epochs,model_file):
#     # model: keras model
#     # training_data: list containing:
#         # training_data[0]: numpy array (data_n x energy_n x 1) for training spectra
#         # training_data[1]: list containing 4 numpy arrays, for each training parameter
#     # validation_data: list containing:
#         # validation_data[0]: numpy array (data_n x energy_n x 1) for validation spectra
#         # validation_data[1]: list containing 4 numpy arrays, for each validation parameter
#     # epochs: int for the number of training epoch
#     # model_file: str for the .keras file to store the model
    
#     # Compile model
#     model.compile(optimizer="adam",
#                   loss={"total_center3":"mae","total_width3":"mae","total_amp3":"mae","total_peak_number3":"mae"},
#                   loss_weights={"total_center3":1.0,"total_width3":10.0,"total_amp3":20.0,"total_peak_number3":2.0},
#                   metrics={"total_center3":"mae","total_width3":"mae","total_amp3":"mae","total_peak_number3":"mae"})

#     # Setups
#     early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
#     model_checkpoint = ModelCheckpoint(model_file, save_best_only=True)
#     reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)
    
#     # Fit model
#     history = model.fit(training_data[0],training_data[1],epochs=epochs, batch_size=512,
#                                       validation_data=(validation_data[0],validation_data[1]),
#                                       callbacks=[model_checkpoint, reduce_lr],shuffle=True,
#                                       verbose=1)
#     return model,history

#%% 2- Compile, setup, train and validate model
def compile_setup_train_validate(model, training_data, validation_data, epochs, model_file):
    # Define the correct layer names based on the four sets of outputs
    losses = {}
    loss_weights = {}
    metrics = {}

    # Manually set the correct names for each set of outputs
    losses['center_set1'] = "mae"
    losses['width_set1'] = "mae"
    losses['amplitude_set1'] = "mae"
    losses['peak_number_set1'] = "mae"
    
    losses['center_set2'] = "mae"
    losses['width_set2'] = "mae"
    losses['amplitude_set2'] = "mae"
    losses['peak_number_set2'] = "mae"
    
    losses['center_set3'] = "mae"
    losses['width_set3'] = "mae"
    losses['amplitude_set3'] = "mae"
    losses['peak_number_set3'] = "mae"
    
    losses['center_set4'] = "mae"
    losses['width_set4'] = "mae"
    losses['amplitude_set4'] = "mae"
    losses['peak_number_set4'] = "mae"

    # Define the loss weights
    loss_weights['center_set1'] = 1.0
    loss_weights['width_set1'] = 10.0
    loss_weights['amplitude_set1'] = 20.0
    loss_weights['peak_number_set1'] = 2.0
    
    loss_weights['center_set2'] = 1.0
    loss_weights['width_set2'] = 10.0
    loss_weights['amplitude_set2'] = 20.0
    loss_weights['peak_number_set2'] = 2.0
    
    loss_weights['center_set3'] = 1.0
    loss_weights['width_set3'] = 10.0
    loss_weights['amplitude_set3'] = 20.0
    loss_weights['peak_number_set3'] = 2.0
    
    loss_weights['center_set4'] = 1.0
    loss_weights['width_set4'] = 10.0
    loss_weights['amplitude_set4'] = 20.0
    loss_weights['peak_number_set4'] = 2.0

    # Define the metrics
    metrics['center_set1'] = "mae"
    metrics['width_set1'] = "mae"
    metrics['amplitude_set1'] = "mae"
    metrics['peak_number_set1'] = "mae"
    
    metrics['center_set2'] = "mae"
    metrics['width_set2'] = "mae"
    metrics['amplitude_set2'] = "mae"
    metrics['peak_number_set2'] = "mae"
    
    metrics['center_set3'] = "mae"
    metrics['width_set3'] = "mae"
    metrics['amplitude_set3'] = "mae"
    metrics['peak_number_set3'] = "mae"
    
    metrics['center_set4'] = "mae"
    metrics['width_set4'] = "mae"
    metrics['amplitude_set4'] = "mae"
    metrics['peak_number_set4'] = "mae"

    # Compile the model
    model.compile(optimizer="adam", loss=losses, loss_weights=loss_weights, metrics=metrics)

    # Set up callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_file, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1)

    # Fit the model
    history = model.fit(
        training_data[0],
        training_data[1],
        epochs=epochs,
        batch_size=512,
        validation_data=(validation_data[0], validation_data[1]),
        callbacks=[model_checkpoint, reduce_lr],
        shuffle=True,
        verbose=1
    )
    
    return model, history



#%% Test model
# def test_model(model,test_data):
#     # model: keras model
#     # test_data: list containing:
#         # test_data[0]: numpy array (data_n x energy_n x 1) for test spectra
#         # test_data[1]: list containing 4 numpy arrays, for each test parameter
    
#     # Apply model to test dataset
#     prediction_0 = model.predict(test_data[0],verbose=1)
#     prediction = np.concatenate(prediction_0,1)
    
#     # Compute MAE
#     test_params = np.stack(test_data[1],1)
#     MAE = np.mean(abs(prediction-test_params),0)
#     MSE = np.mean(abs(prediction-test_params**2),0)
#     RMSE = np.sqrt(MSE)
    
#     # print(f"MAE: {MAE}")
#     # print(f"MSE: {MSE}")
#     # print(f"RMSE: {RMSE}")
    
#     return prediction,MAE,MSE,RMSE

#%% 2- Test model
def test_model(model, test_data):
    # Predict using the model
    predictions = model.predict(test_data[0], verbose=1)
    
    # Concatenate the predictions along axis 1 to get a single (n_samples, 16) array
    predictions = np.concatenate(predictions, axis=1)
    
    # Ensure the ground truth `test_params` contains 16 arrays for comparison
    test_params = [np.reshape(param, (-1, 1)) if len(param.shape) == 1 else param for param in test_data[1]]
    test_params = np.concatenate(test_params, axis=1)
    
    # Ensure the test parameters match the predictions in shape
    if predictions.shape[1] != test_params.shape[1]:
        raise ValueError(f"Mismatch in the number of predictions and test parameters: predictions {predictions.shape[1]}, test parameters {test_params.shape[1]}")
    
    # Compute MAE, MSE, and RMSE for each of the 16 outputs
    MAE = np.mean(np.abs(predictions - test_params), axis=0)
    MSE = np.mean((predictions - test_params) ** 2, axis=0)
    RMSE = np.sqrt(MSE)

    return predictions, MAE, MSE, RMSE


