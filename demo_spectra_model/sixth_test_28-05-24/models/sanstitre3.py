
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_probability as tfp
from tensorflow.keras.utils import plot_model

class MyConvLayer(layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, kernel_divergence_fn, bias_divergence_fn, activation, name=None):
        super(MyConvLayer, self).__init__(name=name)
        self.conv_layer = tfp.layers.Convolution1DFlipout(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_divergence_fn=bias_divergence_fn,
            activation=activation,
        )

    def call(self, inputs):
        return self.conv_layer(inputs)

class MyDenseLayer(layers.Layer):
    def __init__(self, units, kernel_divergence_fn, bias_divergence_fn, activation, name=None):
        super(MyDenseLayer, self).__init__(name=name)
        self.dense_layer = tfp.layers.DenseFlipout(
            units=units,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_divergence_fn=bias_divergence_fn,
            activation=activation,
        )

    def call(self, inputs):
        return self.dense_layer(inputs)

def bayesian_cnn_model(energy_n, kl_divergence_fn, task):
    input_shape = (energy_n, 1)

    strides = 1
    pool_size = 2
    average_pool_layer = layers.AveragePooling1D(pool_size=pool_size)

    output_act = None  # For regression tasks, no activation in the output layer
    prob_act = "softplus"

    input_1 = layers.Input(shape=input_shape, name="input_1")
    conv_1_short = MyConvLayer(
        filters=12,
        kernel_size=5,
        strides=strides,
        padding="same",
        kernel_divergence_fn=kl_divergence_fn,
        bias_divergence_fn=kl_divergence_fn,
        activation=prob_act,
        name="conv_1_short",
    )(input_1)
    conv_1_medium = MyConvLayer(
        filters=12,
        kernel_size=10,
        strides=strides,
        padding="same",
        kernel_divergence_fn=kl_divergence_fn,
        bias_divergence_fn=kl_divergence_fn,
        activation=prob_act,
        name="conv_1_medium",
    )(input_1)
    conv_1_long = MyConvLayer(
        filters=12,
        kernel_size=15,
        strides=strides,
        padding="same",
        kernel_divergence_fn=kl_divergence_fn,
        bias_divergence_fn=kl_divergence_fn,
        activation=prob_act,
        name="conv_1_long",
    )(input_1)

    sublayers = [conv_1_short, conv_1_medium, conv_1_long]
    merged_sublayers = layers.concatenate(sublayers)

    conv_2 = MyConvLayer(
        filters=10,
        kernel_size=5,
        strides=strides,
        padding="valid",
        kernel_divergence_fn=kl_divergence_fn,
        bias_divergence_fn=kl_divergence_fn,
        activation=prob_act,
        name="conv_2",
    )(merged_sublayers)
    conv_3 = MyConvLayer(
        filters=10,
        kernel_size=5,
        strides=strides,
        padding="valid",
        kernel_divergence_fn=kl_divergence_fn,
        bias_divergence_fn=kl_divergence_fn,
        activation=prob_act,
        name="conv_3",
    )(conv_2)
    average_pool_1 = average_pool_layer(conv_3)

    flatten_1 = layers.Flatten(name="flatten1")(average_pool_1)
    drop_1 = layers.Dropout(rate=0.2, name="drop_1")(flatten_1)
    dense_1 = MyDenseLayer(
        units=4000,
        kernel_divergence_fn=kl_divergence_fn,
        bias_divergence_fn=kl_divergence_fn,
        activation=prob_act,
        name="bayesian_dense_1",
    )(drop_1)

    dense2_1 = layers.Dense(100, name="dense2_1")(dense_1)
    batch_norm_1 = layers.BatchNormalization(name="batch_norm_1")(dense2_1)
    leaky_relu_1 = layers.LeakyReLU(name="leaky_relu_1")(batch_norm_1)
    output_1 = layers.Dense(1, activation=output_act, name="output_1")(leaky_relu_1)

    dense2_2 = layers.Dense(100, name="dense2_2")(dense_1)
    batch_norm_2 = layers.BatchNormalization(name="batch_norm_2")(dense2_2)
    leaky_relu_2 = layers.LeakyReLU(name="leaky_relu_2")(batch_norm_2)
    output_2 = layers.Dense(1, activation=output_act, name="output_2")(leaky_relu_2)

    dense2_3 = layers.Dense(100, name="dense2_3")(dense_1)
    batch_norm_3 = layers.BatchNormalization(name="batch_norm_3")(dense2_3)
    leaky_relu_3 = layers.LeakyReLU(name="leaky_relu_3")(batch_norm_3)
    output_3 = layers.Dense(1, activation=output_act, name="output_3")(leaky_relu_3)

    dense2_4 = layers.Dense(100, name="dense2_4")(dense_1)
    batch_norm_4 = layers.BatchNormalization(name="batch_norm_4")(dense2_4)
    leaky_relu_4 = layers.LeakyReLU(name="leaky_relu_4")(batch_norm_4)
    output_4 = layers.Dense(1, activation=output_act, name="output_4")(leaky_relu_4)

    outputs = [output_1, output_2, output_3, output_4]

    return models.Model(inputs=input_1, outputs=outputs, name="BayesianCNN")

# Define the energy_n and other parameters
energy_n = 1121  # Example value for energy_n
kl_divergence_fn = None  # Define KL divergence function appropriately
task = "regression"  # We are performing regression

# Instantiate the BayesianCNN model with all required arguments
bayesian_cnn = bayesian_cnn_model(energy_n, kl_divergence_fn, task)

# Plot the model
#plot_model(bayesian_cnn, to_file='bayesian_cnn_model.png', show_shapes=True)
plot_model(bayesian_cnn, to_file='C:/Users/rsallustre/Documents/XPS_fitting/sixth_test_28-05-24/models/bayesian_cnn_model.png', show_shapes=True)
