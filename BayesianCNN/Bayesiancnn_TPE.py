import tensorflow as tf
import tensorflow_probability as tfp
from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
import sys

import pickle

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

    def compute_output_shape(self, input_shape):
        return self.conv_layer.compute_output_shape(input_shape)

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

def define_bayesian_cnn(params, input_shape):
    average_pool_layer = layers.AveragePooling1D(pool_size=2)
    output_act = None  # For regression tasks, no activation in the output layer

    input_data = layers.Input(shape=input_shape, name="input_1")
    
    conv_1_short = MyConvLayer(
        filters=params['filters_1'],
        kernel_size=params['kernel_size_1'],
        strides=params['strides'],
        padding="same",
        activation=params['activation'],
        name="conv_1_short",
    )(input_data)
    
    conv_1_medium = MyConvLayer(
        filters=params['filters_1'],
        kernel_size=params['kernel_size_1'],
        strides=params['strides'],
        padding="same",
        activation=params['activation'],
        name="conv_1_medium",
    )(input_data)
    
    conv_1_long = MyConvLayer(
        filters=params['filters_1'],
        kernel_size=params['kernel_size_1'] -1,
        strides=params['strides'],
        padding="same",
        activation=params['activation'],
        name="conv_1_long",
    )(input_data)

    merged_sublayers = layers.concatenate([conv_1_short, conv_1_medium, conv_1_long])

    conv_2 = MyConvLayer(
        filters=params['filters_2'],
        kernel_size=params['kernel_size_2'],
        strides=params['strides'],
        padding="same",
        activation=params['activation'],
        name="conv_2",
    )(merged_sublayers)
    
    conv_3 = MyConvLayer(
        filters=params['filters_2'] * 2,
        kernel_size=params['kernel_size_2'],
        strides=params['strides'],
        padding="same",
        activation=params['activation'],
        name="conv_3",
    )(conv_2)
    num_channels =64
    conv_output_tensor = conv_3[0]  # Assuming conv_3 is a tuple (output_tensor, ...)
    conv_output_shape = tf.shape(conv_output_tensor).numpy() # Get the shape as a list
    sequence_length = conv_output_shape[1]  # Assuming sequence length is the second dimension

# Reshape conv_output_tensor to (None, sequence_length, num_channels)
    reshaped_conv_3 = layers.Reshape((sequence_length, num_channels))(conv_output_tensor)

# Then continue with your model construction
    average_pool_1 = average_pool_layer(reshaped_conv_3)


    flatten_1 = layers.Flatten(name="flatten1")(average_pool_1)
    drop_1 = layers.Dropout(rate=params['dropout_rate'], name="drop_1")(flatten_1)

    dense2_1 = layers.Dense(512, name="dense2_1")(drop_1)
    batch_norm_1 = layers.BatchNormalization(name="batch_norm_1")(dense2_1)
    leaky_relu_1 = layers.LeakyReLU(name="leaky_relu_1")(batch_norm_1)
    total_center3 = layers.Dense(1, activation=output_act, name="total_center3")(leaky_relu_1)

    dense2_2 = layers.Dense(512, name="dense2_2")(drop_1)
    batch_norm_2 = layers.BatchNormalization(name="batch_norm_2")(dense2_2)
    leaky_relu_2 = layers.LeakyReLU(name="leaky_relu_2")(batch_norm_2)
    total_width3 = layers.Dense(1, activation=output_act, name="total_width3")(leaky_relu_2)

    dense2_3 = layers.Dense(512, name="dense2_3")(drop_1)
    batch_norm_3 = layers.BatchNormalization(name="batch_norm_3")(dense2_3)
    leaky_relu_3 = layers.LeakyReLU(name="leaky_relu_3")(batch_norm_3)
    total_amp3 = layers.Dense(1, activation=output_act, name="total_amp3")(leaky_relu_3)

    dense2_4 = layers.Dense(512, name="dense2_4")(drop_1)
    batch_norm_4 = layers.BatchNormalization(name="batch_norm_4")(dense2_4)
    leaky_relu_4 = layers.LeakyReLU(name="leaky_relu_4")(batch_norm_4)
    total_peak_number3 = layers.Dense(1, activation=output_act, name="total_peak_number3")(leaky_relu_4)

    outputs = [total_center3, total_width3, total_amp3, total_peak_number3]
    model = Model(inputs=input_data, outputs=outputs, name="BayesianCNN")

    return model

# # Load database
# main_data_folder = "C:/Users/rsallustre/Documents/XPS_fitting/"
# session_name = "Ninth_test_19-06-24"
# data_folder = main_data_folder + session_name + "/"
# database_folder = data_folder + "Database/"

# # Load training data from pickle file
# with open(database_folder + "Training_database.pkl", 'rb') as f:
#     train_data = pickle.load(f)

# # Load validation data from pickle file
# with open(database_folder + "Validation_database.pkl", 'rb') as f:
#     val_data = pickle.load(f)

# # Assuming your data structure in the pickle file is [X_train, y_train]
# X_train = train_data[0]
# y_train = train_data[1]

# # Assuming your data structure in the validation pickle file is [X_val, y_val]
# X_val = val_data[0]
# y_val = val_data[1]

# # Ensure correct shape and type of input data
# X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
# X_val = np.expand_dims(X_val, axis=-1)      # Add channel dimension
# X_train = X_train.astype('float32')         # Convert to float32 if needed
# X_val = X_val.astype('float32')             # Convert to float32 if needed
num_samples = 1000
sequence_length = 100
num_channels = 1
num_classes = 1

# Generate synthetic training data
X_train = np.random.rand(num_samples, sequence_length, num_channels)
y_train = np.random.rand(num_samples, num_classes)

# Generate synthetic validation data
X_val = np.random.rand(num_samples // 5, sequence_length, num_channels)
y_val = np.random.rand(num_samples // 5, num_classes)

# Function to define objective for Hyperopt
def objective(params):
    model = define_bayesian_cnn(params, X_train.shape[1:])
    optimizer = {
        'adam': tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=params['learning_rate']),
        'sgd': tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])
    }[params['optimizer']]

    model.compile(optimizer=optimizer, loss='mse')

    # Use the loaded X_train, y_train, X_val, y_val directly
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(patience=10),
            ReduceLROnPlateau(patience=5)
        ],
        verbose=0
    )

    val_loss = min(history.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK}

# Hyperparameter space definition
    

space = {
    'filters_1': hp.choice('filters_1', [32, 64, 128]),
    'filters_2': hp.choice('filters_2', [64, 128, 256]),
    'kernel_size_1': hp.choice('kernel_size_1', [3, 5, 7]),
    'kernel_size_2': hp.choice('kernel_size_2', [3, 5, 7]),
    'strides': hp.choice('strides', [1, 2]),
    'activation': hp.choice('activation', ['relu', 'tanh', 'sigmoid']),
    'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.5),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'sgd']),
    'learning_rate': hp.loguniform('learning_rate', -4, -2),
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'epochs': 50
}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

print(best)
