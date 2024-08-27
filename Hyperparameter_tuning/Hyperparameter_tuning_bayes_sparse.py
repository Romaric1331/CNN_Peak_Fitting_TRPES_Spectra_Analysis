import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, LeakyReLU, Conv1D, MaxPooling1D, AveragePooling1D, Multiply, Add, Concatenate, GlobalAveragePooling1D, Dense, Activation, Reshape
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
class BayesianConv1D(tf.keras. Layer):
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

def define_bayesian_sparse_densenet(energy_n, filters, kernel_size,strides, learning_rate, kl_weight, dense_units):
    input_data = Input(shape=(energy_n, 1))
    r = 16
    Cf = 0.5
    shortcut = 0
    se = 0
    # Sparse dense block with no se block

    x = BayesianConv1D(32, 4, strides=2, padding="same", kernel_initializer="he_normal")(input_data)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = BayesianConv1D(32, 4, strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = BayesianConv1D(32, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = MaxPooling1D(3, strides=2)(x)  # divide by 2
    
    shortcut_dense = x
    shortcut = x
    shortcut = BayesianConv1D(64, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)


# ------------------ first layer-1
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x = BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se =  GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(64 // r, kernel_initializer="he_normal")(se)  # FC
    se =  LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(64, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid

    x = Multiply()([x, se])  # Scale
    x =  Add()([x, shortcut])  # x.shape = (100,256)
    

# ----------------- first layer-2
    shortcut = x
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(64, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se =  GlobalAveragePooling1D()(x)
    se = Dense(64 // r, kernel_initializer="he_normal")(se)
    se =  LeakyReLU(alpha=0.01)(se)
    se = Dense(64, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    # se= Reshape([1,100])(se)
    
    x = Multiply()([x, se])
    x =  Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])  # Added layer into the concatenate
    
    # se =  GlobalAveragePooling1D()(x)
    # se = Dense(96 // r, kernel_initializer = 'he_normal')(se)
    # se =  LeakyReLU(alpha = 0.01)(se)
    # se = Dense(96, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale


# -----------------------------transition layer
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(int(96*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # 나누기 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)  # overlapped pooling
    
    se =  GlobalAveragePooling1D()(x)
    se = Dense(48 // r, kernel_initializer="he_normal")(se)
    se =  LeakyReLU(alpha=0.01)(se)
    se = Dense(48, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale


# --------------------------------------
# ----------------- second layer-1
    shortcut_dense = x
    
    shortcut = x
    shortcut = BayesianConv1D(128, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se =  GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(128 // r, kernel_initializer="he_normal")(se)  # FC
    se =  LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(128, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x =  Add()([x, shortcut])
    
    
    # ----------------- second layer-2
    shortcut = x
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(128,  3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(128, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se =  GlobalAveragePooling1D()(x)
    se = Dense(128 // r, kernel_initializer="he_normal")(se)
    se =  LeakyReLU(alpha=0.01)(se)
    se = Dense(128, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])
    x =  Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])
    
    # se =  GlobalAveragePooling1D()(x)
    # se = Dense(176 // r, kernel_initializer = 'he_normal')(se)
    # se =  LeakyReLU(alpha = 0.01)(se)
    # se = Dense(176, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # transition layer---------------------------------
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(int(176*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)
    
    se =  GlobalAveragePooling1D()(x)
    se = Dense(88 // r, kernel_initializer="he_normal")(se)
    se =  LeakyReLU(alpha=0.01)(se)
    se = Dense(88, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # ----------------- third layer-1
    shortcut_dense = x
    
    shortcut = x
    shortcut = BayesianConv1D(256, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se =  GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(256 // r, kernel_initializer="he_normal")(se)  # FC
    se =  LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(256, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid

    x = Multiply()([x, se])  # Scale
    x =  Add()([x, shortcut])
    
    
    # ----------------- third layer-2
    shortcut = x
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(256, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se =  GlobalAveragePooling1D()(x)
    se = Dense(256 // r, kernel_initializer="he_normal")(se)
    se =  LeakyReLU(alpha=0.01)(se)
    se = Dense(256, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])
    x =  Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])
    
    # se =  GlobalAveragePooling1D()(x)
    # se = Dense(344 // r, kernel_initializer = 'he_normal')(se)
    # se =  LeakyReLU(alpha = 0.01)(se)
    # se = Dense(344, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # transition layer---------------------------------
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(int(334*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)
    
    se =  GlobalAveragePooling1D()(x)
    se = Dense(167 // r, kernel_initializer="he_normal")(se)
    se =  LeakyReLU(alpha=0.01)(se)
    se = Dense(167, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # ----------------- four layer-1
    shortcut_dense = x
    shortcut = x
    shortcut = BayesianConv1D(512, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se =  GlobalAveragePooling1D()(x)  # global pooling
    se = Dense(512 // r, kernel_initializer="he_normal")(se)  # FC
    se =  LeakyReLU(alpha=0.01)(se)  # ReLU
    se = Dense(512, kernel_initializer="he_normal")(se)  # FC
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])  # Scale
    x =  Add()([x, shortcut])
    
    
    # ----------------- four layer-2
    shortcut = x
    
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    x =  BatchNormalization()(x)  # 786,994
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(512, 3, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    
    se =  GlobalAveragePooling1D()(x)
    se = Dense(512 // r, kernel_initializer="he_normal")(se)
    se =  LeakyReLU(alpha=0.01)(se)
    se = Dense(512, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    
    x = Multiply()([x, se])
    x =  Add()([x, shortcut])
    x = Concatenate()([x, shortcut_dense])
    
    # se =  GlobalAveragePooling1D()(x)
    # se = Dense(679 // r, kernel_initializer = 'he_normal')(se)
    # se =  LeakyReLU(alpha = 0.01)(se)
    # se = Dense(679, kernel_initializer = 'he_normal')(se)
    # se = Activation('sigmoid')(se) # Sigmoid
    
    # x = Multiply()([x,se])  # scale
    
    
    # transition layer---------------------------------
    x =  BatchNormalization()(x)
    x =  LeakyReLU(alpha=0.01)(x)
    x = BayesianConv1D(int(679*Cf), 1, strides=1, padding="same", kernel_initializer="he_normal")(x)  # divide by 2
    x = AveragePooling1D(3, padding="same", strides=2)(x)
    
    se =  GlobalAveragePooling1D()(x)
    se = Dense(339 // r, kernel_initializer="he_normal")(se)
    se =  LeakyReLU(alpha=0.01)(se)
    se = Dense(339, kernel_initializer="he_normal")(se)
    se = Activation("sigmoid")(se)  # Sigmoid
    x = Multiply()([x, se])  # Scale
    
    
    # --------------------------------------
    # --------------------------------------
    x =  GlobalAveragePooling1D()(x)
    
    
    total_center1 = Dense(100, name="total_center1", kernel_initializer="he_normal")(x)
    center_Batchnormalization = BatchNormalization()(total_center1)
    total_center1_act =  LeakyReLU(alpha=0.01)(center_Batchnormalization)
    total_center3 = Dense(1, activation="linear", name="total_center3", kernel_initializer="he_normal")(total_center1_act)
    
    total_width1 = Dense(100, name="total_width1", kernel_initializer="he_normal")(x)
    width_Batchnormalization = BatchNormalization()(total_width1)
    total_width1_act =  LeakyReLU(alpha=0.01)(width_Batchnormalization)
    total_width3 = Dense(1, activation="linear", name="total_width3", kernel_initializer="he_normal")(total_width1_act)
    
    total_amp1 = Dense(100, name="total_amp1", kernel_initializer="he_normal")(x)
    amp_Batchnormalization = BatchNormalization()(total_amp1)
    total_amp1_act =  LeakyReLU(alpha=0.01)(amp_Batchnormalization)
    total_amp3 = Dense(1, activation="linear", name="total_amp3", kernel_initializer="he_normal")(total_amp1_act)
    
    total_peak_number1 = Dense(100, name="total_peak_number1", kernel_initializer="he_normal")(x)
    peak_number_Batchnormalization = BatchNormalization()(total_peak_number1)
    total_peak_number1_act = LeakyReLU(alpha=0.01)(peak_number_Batchnormalization)
    total_peak_number3 = Dense(1, activation="linear", name="total_peak_number3", kernel_initializer="he_normal")(total_peak_number1_act)
    
    
    model = Model(inputs=input_data,outputs=[total_center3, total_width3, total_amp3, total_peak_number3])
    

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
    model = define_bayesian_sparse_densenet(energy_n, filters, strides, kernel_size, learning_rate, kl_weight, dense_units)

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
fig1.write_html('optimization_history.html', auto_open=True)
fig2.write_html('parallel_coordinate.html', auto_open=True)
fig3.write_html('slice_plot.html', auto_open=True)
fig4.write_html('param_importances.html', auto_open=True)
# Specify the directory where you want to save the files
output_directory = "C:/Users/rsallustre/Documents/GitHub/IA_spectro/Hyperparameter_tuning/Bayesian_sparsedensenet/"

# Ensure the directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Save HTML files
fig1.write_html(output_directory + 'optimization_history_Bayes_sparse.html', auto_open=False)
fig2.write_html(output_directory + 'parallel_coordinate_Bayes_sparse.html', auto_open=False)
fig3.write_html(output_directory + 'slice_plot_Bayes_sparse.html', auto_open=False)
fig4.write_html(output_directory + 'param_importances_Bayes_sparse.html', auto_open=False)

# Save the best value to a text file
with open(output_directory + 'study_Bayes_sparse_summary.txt', 'w') as f:
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

