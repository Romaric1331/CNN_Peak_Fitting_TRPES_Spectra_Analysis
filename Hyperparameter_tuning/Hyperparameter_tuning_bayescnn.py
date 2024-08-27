# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:00:21 2024

@author: rsallustre
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, concatenate, Flatten, Dropout, LeakyReLU, Conv1D, MaxPooling1D, AveragePooling1D, Multiply, Add, Concatenate, GlobalAveragePooling1D, Dense, Activation, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import optuna
import pickle
import matplotlib.pyplot as plt
import plotly
from plotly.io import to_image
import io, os
#%% Function def
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
    
  
def define_bayesian_cnn(energy_n, filters, strides, kernel_size, learning_rate, kl_weight, dense_units):
    strides = 1
    pool_size = 2
    average_pool_layer = AveragePooling1D(pool_size=pool_size)
    output_act = None  # For regression tasks, no activation in the output layer
    prob_act = "softplus"

    input_data = Input(shape=(energy_n, 1), name="input_1")
    conv_1_short = MyConvLayer(
        filters=32,
        kernel_size=4,
        strides=strides,
        padding="same",
        activation=prob_act,
        name="conv_1_short",
    )(input_data)
    conv_1_medium = MyConvLayer(
        filters=32,
        kernel_size=4,
        strides=strides,
        padding="same",
        activation=prob_act,
        name="conv_1_medium", 
    )(input_data)
    conv_1_long = MyConvLayer(
        filters=32,
        kernel_size=3,
        strides=strides,
        padding="same",
        activation=prob_act,
        name="conv_1_long",
    )(input_data)

    merged_sublayers = concatenate([conv_1_short, conv_1_medium, conv_1_long])

    conv_2 = MyConvLayer(
        filters=64,
        kernel_size=3,
        strides=strides,
        padding="same",
        activation=prob_act,
        name="conv_2",
    )(merged_sublayers)
    conv_3 = MyConvLayer(
        filters=128,
        kernel_size=3,
        strides=strides,
        padding="same",
        activation=prob_act,
        name="conv_3",
    )(conv_2)
    average_pool_1 = average_pool_layer(conv_3)

    flatten_1 = Flatten(name="flatten1")(average_pool_1)
    drop_1 = Dropout(rate=0.2, name="drop_1")(flatten_1)

    dense2_1 = Dense(512, name="dense2_1")(drop_1)
    batch_norm_1 = BatchNormalization(name="batch_norm_1")(dense2_1)
    leaky_relu_1 = LeakyReLU(name="leaky_relu_1")(batch_norm_1)
    total_center3 = Dense(1, activation=output_act, name="total_center3")(leaky_relu_1)

    dense2_2 = Dense(512, name="dense2_2")(drop_1)
    batch_norm_2 = BatchNormalization(name="batch_norm_2")(dense2_2)
    leaky_relu_2 = LeakyReLU(name="leaky_relu_2")(batch_norm_2)
    total_width3 = Dense(1, activation=output_act, name="total_width3")(leaky_relu_2)

    dense2_3 = Dense(512, name="dense2_3")(drop_1)
    batch_norm_3 = BatchNormalization(name="batch_norm_3")(dense2_3)
    leaky_relu_3 = LeakyReLU(name="leaky_relu_3")(batch_norm_3)
    total_amp3 = Dense(1, activation=output_act, name="total_amp3")(leaky_relu_3)

    dense2_4 = Dense(512, name="dense2_4")(drop_1)
    batch_norm_4 = BatchNormalization(name="batch_norm_4")(dense2_4)
    leaky_relu_4 = LeakyReLU(name="leaky_relu_4")(batch_norm_4)
    total_peak_number3 = Dense(1, activation=output_act, name="total_peak_number3")(leaky_relu_4)

    outputs = [total_center3, total_width3, total_amp3, total_peak_number3]
    model = Model(inputs=input_data, outputs=outputs, name="BayesianCNN")

    return model
#%% Load database
# Load the training and validation databases from .pkl files
main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting/"
session_name = "Twelth_test_11-07-24"

data_folder = main_data_folder + session_name + "/"
database_folder = data_folder + "Database/"

# Load the training dataset
with open(database_folder + "Training_database.pkl", 'rb') as f:
    energy_range, train_peak_label, train_peak, train_peak_param = pickle.load(f)
train_n = len(train_peak_label)
energy_n = len(energy_range)

# Load the validation dataset
with open(database_folder + "Validation_database.pkl", 'rb') as f:
    energy_range, val_peak_label, val_peak, val_peak_param = pickle.load(f)
val_n = len(val_peak_label)

# Load the test dataset
with open(database_folder + "Test_database.pkl", 'rb') as f:
    energy_range, test_peak_label, test_peak, test_peak_param = pickle.load(f)
test_n = len(test_peak_label)

# Prepare the data for the model
X_train = np.array(train_peak).reshape((train_n, energy_n, 1)).astype('float32')
X_val = np.array(val_peak).reshape((val_n, energy_n, 1)).astype('float32')
X_test = np.array(test_peak).reshape((test_n, energy_n, 1)).astype('float32')

# Flatten the sequences and determine the maximum length
flat_train_peak_param = [np.array(seq).flatten() for seq in train_peak_param]
flat_val_peak_param = [np.array(seq).flatten() for seq in val_peak_param]
flat_test_peak_param = [np.array(seq).flatten() for seq in test_peak_param]

max_length_y_train = max(len(seq) for seq in flat_train_peak_param)
max_length_y_val = max(len(seq) for seq in flat_val_peak_param)
max_length_y_test = max(len(seq) for seq in flat_test_peak_param)

# Pad all sequences to the maximum length
y_train = np.array([np.pad(seq, (0, max_length_y_train - len(seq)), mode='constant') for seq in flat_train_peak_param]).astype('float32')
y_val = np.array([np.pad(seq, (0, max_length_y_val - len(seq)), mode='constant') for seq in flat_val_peak_param]).astype('float32')
y_test = np.array([np.pad(seq, (0, max_length_y_test - len(seq)), mode='constant') for seq in flat_test_peak_param]).astype('float32')

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    filters = trial.suggest_int('filters', 32, 64)  # Reduce the number of filters
    kernel_size = trial.suggest_int('kernel_size', 3, 5)  # Smaller kernel size
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)  # Smaller initial learning rate
    kl_weight = trial.suggest_float('kl_weight', 1e-3, 1e-1, log=True)  # Adjust KL weight
    batch_size = trial.suggest_categorical('batch_size', [16, 32])  # Smaller batch sizes
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)  # Weight decay for regularization
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)  # Dropout rate
    beta1 = trial.suggest_float('beta1', 0.8, 0.99)  # Adam optimizer beta1
    beta2 = trial.suggest_float('beta2', 0.99, 0.999)  # Adam optimizer beta2
    epsilon = trial.suggest_float('epsilon', 1e-7, 1e-5, log=True)  # Adam optimizer epsilon
    epochs = trial.suggest_int('epochs', 40, 100)  # Number of epochs
    strides = trial.suggest_int('strides', 1, 3)  # Strides for the convolution layer
    dense_units = trial.suggest_int('dense_units', 32, 128)  # Units for the dense layer

    # Build the model with the suggested hyperparameters
    model = define_bayesian_cnn(energy_n, filters, strides, kernel_size, learning_rate, kl_weight, dense_units)

    # Compile the model with the specified optimizer and metrics
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon, decay=weight_decay)
    # Use MAE as the loss function
    loss = 'mae'

    model.compile(optimizer=optimizer, loss=loss)

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=19, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
        optuna.integration.TFKerasPruningCallback(trial, 'val_loss')
    ]

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs
                        , batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks)

    # Evaluate the model on the test dataset
    eval_results = model.evaluate(X_test, y_test)
    
    # Return the test loss
    if isinstance(eval_results, list) or isinstance(eval_results, tuple):
        return eval_results[0]  # Assuming the first element is the loss
    else:
        return eval_results  # Return the single float value

# Create a study object and specify the direction is 'minimize'.
study = optuna.create_study(direction='minimize')

# Optimize the study, the objective function is passed in as the first argument.
study.optimize(objective, n_trials=100)  # Reduce the number of trials

# Print the best hyperparameters
print("Best hyperparameters:")
print(study.best_params)
print("Best loss:")
print(study.best_value)
# Visualize the optimization history
fig1 = optuna.visualization.plot_optimization_history(study)


# Visualize the parallel coordinate plot
fig2 = optuna.visualization.plot_parallel_coordinate(study)


# Visualize the slice plot
fig3 = optuna.visualization.plot_slice(study)


# Visualize the parameter importances
fig4 = optuna.visualization.plot_param_importances(study)
# Render the plots in an external browser

# Specify the directory where you want to save the files
output_directory = "C:/Users/rsallustre/Documents/GitHub/IA_spectro/Hyperparameter_tuning/BayesianCNN/"

# Ensure the directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Save HTML files
fig1.write_html(output_directory + 'optimization_histor_Bayescnn.html', auto_open=True)
fig2.write_html(output_directory + 'parallel_coordinate_Bayescnn.html', auto_open=True)
fig3.write_html(output_directory + 'slice_plot_Bayescnn.html', auto_open=True)
fig4.write_html(output_directory + 'param_importances_Bayescnn.html', auto_open=True)
# Save the best value to a text file
with open(output_directory + 'study_Bayes_cnn_summary.txt', 'w') as f:
    f.write(f"Best loss: {study.best_value}\n")
    f.write(f"Best hyperparameters:\n")
    for key, value in study.best_params.items():
        f.write(f"{key}: {value}\n")
# # Convert the figures to static images and display them
# figs = [fig1, fig2, fig3, fig4]
# for i, fig in enumerate(figs):
#     img_bytes = to_image(fig, format='png')
#     img_arr = np.array(bytearray(img_bytes), dtype=np.uint8)
#     img = plt.imread(io.BytesIO(img_arr))

#     plt.figure(i)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()

