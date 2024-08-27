import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Multiply, Add, Concatenate, AveragePooling1D, GlobalAveragePooling1D
from tensorflow.keras import layers
import tensorflow_probability as tfp
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
def define_sparse_densenet(energy_n):
    input_data = Input(shape=(energy_n, 1))
    r = 16
    Cf = 0.5
# Sparse dense block with no se block


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
    shortcut = layers.Conv1D(256, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
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
    shortcut = layers.Conv1D(512, 1, strides=1, padding="valid", kernel_initializer="he_normal")(shortcut)
    
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

    
    return model
energy_n = 401
model = define_sparse_densenet(energy_n)
model.summary()
model = define_sparse_densenet(401)  # Assuming energy_n is 401
tf.keras.utils.plot_model(model, to_file='model.pdf', show_shapes=True, show_layer_names=True)
import os
import tensorflow as tf

# Define the output folder
output_folder = "C:/Users/rsallustre/Documents/GitHub/IA_spectro/BayesianCNN/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define the full path for the output file
output_file = os.path.join(output_folder, "bayes_sparse_model_architecture.pdf")

# Save the model plot as a PDF
tf.keras.utils.plot_model(
    model,
    to_file=output_file,
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96
)

print(f"Model architecture saved to: {output_file}")


