import pickle
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping

tfd = tfp.distributions
tfpl = tfp.layers

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / wid)

def generate_random_spectrum(num_points=401, num_spectra=1000):
    x = np.linspace(0, 400, num_points)
    y_list = []
    for _ in range(num_spectra):
        y = np.random.normal(0, 0.1, num_points)  # Add some noise
        for _ in range(np.random.randint(3, 6)):  # 3 to 5 random peaks
            amp = np.random.uniform(0.5, 2.0)
            cen = np.random.uniform(50, 350)
            wid = np.random.uniform(100, 1000)
            y += gaussian(x, amp, cen, wid)
        y = np.maximum(y, 0)  # Ensure all y values are positive
        y_list.append(y)
    return x, np.array(y_list)

# Generate training and validation data
X_train, y_train = generate_random_spectrum(num_spectra=800)
X_val, y_val = generate_random_spectrum(num_spectra=200)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# Plot a random spectrum from the training set
random_index = np.random.randint(0, len(y_train))
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train[random_index])
plt.title('Random 1D Spectrum from Training Set')
plt.xlabel('Energy')
plt.ylabel('Intensity')
plt.show()

# Save arrays for later use
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)

def define_bayesian_cnn(params, input_shape):
    filters_1 = params['filters_1']
    filters_2 = params['filters_2']
    kernel_size_1 = params['kernel_size_1']
    kernel_size_2 = params['kernel_size_2']
    strides = params['strides']
    activation = params['activation']
    dropout_rate = params['dropout_rate']
    optimizer = params['optimizer']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']

    def prior(kernel_size, dtype=None):
        return lambda *args, **kwargs: tfd.Normal(loc=0., scale=1.)

    def posterior(kernel_size, dtype=None):
        n = np.prod(kernel_size)
        return lambda *args, **kwargs: tfp.layers.util.default_mean_field_normal_fn(
        loc_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.1),
        untransformed_scale_initializer=tf.keras.initializers.RandomNormal(mean=-3., stddev=0.1)
    )(n, *args, **kwargs)



    input_data = layers.Input(shape=input_shape, name="input_1")
    
    # Reshape the 1D input to 3D for Conv1D layers
    reshaped_input = layers.Reshape((input_shape[0], 1))(input_data)

    conv_1_short = tfpl.Convolution1DFlipout(
    filters=filters_1, kernel_size=kernel_size_1, strides=strides, padding="same",
    activation=activation, name="conv_1_short",
    kernel_prior_fn=prior, kernel_posterior_fn=posterior
    )(input_data)
    conv_1_medium = tfpl.Convolution1DFlipout(
        filters=filters_1, kernel_size=kernel_size_1, strides=strides, padding="same",
        activation=activation, name="conv_1_medium",
        kernel_prior_fn=prior, kernel_posterior_fn=posterior
    )(reshaped_input)
    conv_1_long = tfpl.Convolution1DFlipout(
        filters=filters_1, kernel_size=kernel_size_2, strides=strides, padding="same",
        activation=activation, name="conv_1_long",
        kernel_prior_fn=prior, kernel_posterior_fn=posterior
    )(reshaped_input)

    merged_sublayers = layers.concatenate([conv_1_short, conv_1_medium, conv_1_long])

    conv_2 = tfpl.Convolution1DFlipout(
        filters=filters_2, kernel_size=kernel_size_2, strides=strides, padding="same",
        activation=activation, name="conv_2",
        kernel_prior_fn=prior, kernel_posterior_fn=posterior
    )(merged_sublayers)
    
    conv_3 = tfpl.Convolution1DFlipout(
        filters=filters_2, kernel_size=kernel_size_2, strides=strides, padding="same",
        activation=activation, name="conv_3",
        kernel_prior_fn=prior, kernel_posterior_fn=posterior
    )(conv_2)
    
    average_pool_1 = layers.AveragePooling1D(pool_size=2)(conv_3)
    flatten_1 = layers.Flatten(name="flatten1")(average_pool_1)
    drop_1 = layers.Dropout(rate=dropout_rate, name="drop_1")(flatten_1)

    dense2_1 = tfpl.DenseFlipout(512, activation=activation, name="dense2_1",
                                 kernel_prior_fn=prior, kernel_posterior_fn=posterior)(drop_1)
    batch_norm_1 = layers.BatchNormalization(name="batch_norm_1")(dense2_1)
    leaky_relu_1 = layers.LeakyReLU(name="leaky_relu_1")(batch_norm_1)

    dense2_2 = tfpl.DenseFlipout(512, activation=activation, name="dense2_2",
                                 kernel_prior_fn=prior, kernel_posterior_fn=posterior)(leaky_relu_1)
    batch_norm_2 = layers.BatchNormalization(name="batch_norm_2")(dense2_2)
    leaky_relu_2 = layers.LeakyReLU(name="leaky_relu_2")(batch_norm_2)

    dense2_3 = tfpl.DenseFlipout(512, activation=activation, name="dense2_3",
                                 kernel_prior_fn=prior, kernel_posterior_fn=posterior)(leaky_relu_2)
    batch_norm_3 = layers.BatchNormalization(name="batch_norm_3")(dense2_3)
    leaky_relu_3 = layers.LeakyReLU(name="leaky_relu_3")(batch_norm_3)

    dense2_4 = tfpl.DenseFlipout(512, activation=activation, name="dense2_4",
                                 kernel_prior_fn=prior, kernel_posterior_fn=posterior)(leaky_relu_3)
    batch_norm_4 = layers.BatchNormalization(name="batch_norm_4")(dense2_4)
    leaky_relu_4 = layers.LeakyReLU(name="leaky_relu_4")(batch_norm_4)

    total_center3 = tfpl.DenseFlipout(1, activation=None, name="total_center3",
                                      kernel_prior_fn=prior, kernel_posterior_fn=posterior)(leaky_relu_4)
    total_width3 = tfpl.DenseFlipout(1, activation=None, name="total_width3",
                                     kernel_prior_fn=prior, kernel_posterior_fn=posterior)(leaky_relu_4)
    total_amp3 = tfpl.DenseFlipout(1, activation=None, name="total_amp3",
                                   kernel_prior_fn=prior, kernel_posterior_fn=posterior)(leaky_relu_4)
    total_peak_number3 = tfpl.DenseFlipout(1, activation=None, name="total_peak_number3",
                                           kernel_prior_fn=prior, kernel_posterior_fn=posterior)(leaky_relu_4)

    model = Model(inputs=input_data, outputs=[total_center3, total_width3, total_amp3, total_peak_number3], name="BayesianCNN")

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    model.compile(optimizer=opt, loss='mse')
    return model

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
}

def objective(params):
    input_shape = (401, 1)  # Shape is now (401, 1) for Conv1D layers
    model = define_bayesian_cnn(params, input_shape)
    
    # Reshape X_train and X_val
    X_train_reshaped = X_train.reshape(-1, 401, 1)
    X_val_reshaped = X_val.reshape(-1, 401, 1)
    
    history = model.fit(
        X_train_reshaped,
        [y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3]],
        epochs=50,
        batch_size=params['batch_size'],
        validation_data=(X_val_reshaped, [y_val[:, 0], y_val[:, 1], y_val[:, 2], y_val[:, 3]]),
        callbacks=[
            EarlyStopping(patience=10),
            ReduceLROnPlateau(patience=5)
        ],
        verbose=0
    )
    val_loss = min(history.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK}


trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

print(best)
