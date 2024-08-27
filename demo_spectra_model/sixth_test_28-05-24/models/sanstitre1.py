import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers, models
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import plot_model

class EmptyModel(models.Model):
    """Base Model class."""

    def __init__(
        self,
        inputs,
        outputs,
        inputshape,
        num_classes,
        no_of_inputs=1,
        name="New_Model",
    ):
        """
        Intialize emppy keras model.

        Aside from the inputs and outputs for the instantion of the
        Model class from Keras, the EmptyModel class also gets as
        paramters the input shape of the data, the no. of classes
        of the labels as well as how many times the input shall be
        used.

        Parameters
        ----------
        inputs : keras.Input object or list of keras.Input objects.
            Inputs for the instantion of the Model class from Keras.
        outputs : Outputs of the last layer.
            Outputs for the instantion of the Model class from Keras.
        inputshape : ndarray
            Shape of the features of the training data set.
        num_classes : ndarray
            Shape of the labels of the training data set.
        no_of_inputs : int, optional
            Number of times the input shall be used in the Model.
            The default is 1.
        name : str, optional
            Name of the model.
            The default is "New_Model".

        Returns
        -------
        None.

        """
        self.inputshape = inputshape
        self.num_classes = num_classes
        self.no_of_inputs = no_of_inputs

        super().__init__(inputs=inputs, outputs=outputs, name=name)

    def get_config(self):
        """
        Overwrite get_config method.

        For serialization, all input paramters of the model are added to
        the get_config method from the keras.Model class.

        Returns
        -------
        config : dict
            Configuration of the model.

        """
        # For serialization with "custom_objects"
        config = super().get_config()
        config["inputshape"] = self.inputshape
        config["num_classes"] = self.num_classes
        config["no_of_inputs"] = self.no_of_inputs

        return config

class BayesianCNN(EmptyModel):
    """
    A CNN with three convolutional layers of different kernel size at
    the beginning. Works well for learning across scales.

    This is to be used for regression on all labels. -> sigmoid
    activation in the last layer.
    """

    def __init__(
        self,
        inputshape,
        num_classes,
        kl_divergence_fn,
        task,
    ):
        if len(inputshape) == 2:
            conv_layer = tfp.layers.Convolution1DFlipout
            strides = 1
            average_pool_layer = layers.AveragePooling1D
        elif len(inputshape) == 3:
            conv_layer = tfp.layers.Convolution2DFlipout
            strides = (1, 1)
            average_pool_layer = layers.AveragePooling2D

        if task == "regression":
            if num_classes == 1:
                output_act = None
            else:
                output_act = None  # "sigmoid"
        elif task == "classification":
            output_act = "softmax"

        prob_act = "softplus"  # Define prob_act here

        conv_1_short_input = layers.Input(shape=input_shape, name="conv_1_short_input")
        self.conv_1_short = conv_layer(
            filters=12,
            kernel_size=5,
            strides=strides,
            padding="same",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,  # Use prob_act here
            name="conv_1_short",
        )(conv_1_short_input)
        
        conv_1_medium_input = layers.Input(shape=input_shape, name="conv_1_medium_input")
        self.conv_1_medium = conv_layer(
            filters=12,
            kernel_size=10,
            strides=strides,
            padding="same",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,  # Use prob_act here
            name="conv_1_medium",
        )(conv_1_medium_input)

        conv_1_long_input = layers.Input(shape=input_shape, name="conv_1_long_input")
        self.conv_1_long = conv_layer(
            filters=12,
            kernel_size=15,
            strides=strides,
            padding="same",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,  # Use prob_act here
            name="conv_1_long",
        )(conv_1_long_input)

        sublayers = [
            self.conv_1_short,
            self.conv_1_medium,
            self.conv_1_long,
        ]
        merged_sublayers = layers.concatenate(sublayers)

        self.conv_2 = conv_layer(
            filters=10,
            kernel_size=5,
            strides=strides,
            padding="valid",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,  # Use prob_act here
            name="conv_2",
        )(merged_sublayers)
        self.conv_3 = conv_layer(
            filters=10,
            kernel_size=5,
            strides=strides,
            padding="valid",
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,  # Use prob_act here
            name="conv_3",
        )(self.conv_2)
        self.average_pool_1 = average_pool_layer(name="average_pool_1")(self.conv_3)

        self.flatten_1 = layers.Flatten(name="flatten1")(self.average_pool_1)
        self.drop_1 = layers.Dropout(rate=0.2, name="drop_1")(self.flatten_1)
        self.dense_1 = tfp.layers.DenseFlipout(
            units=4000,
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=prob_act,  # Use prob_act here
            name="dense_1",
        )(self.flatten_1)

        self.dense_2 = tfp.layers.DenseFlipout(
            units=num_classes,
            kernel_divergence_fn=kl_divergence_fn,
            bias_divergence_fn=kl_divergence_fn,
            activation=output_act,
            name="dense_2",
        )(self.dense_1)

        no_of_inputs = len(sublayers)

        super().__init__(
            inputs=[conv_1_short_input, conv_1_medium_input, conv_1_long_input],
            outputs=self.dense_2,
            inputshape=inputshape,
            num_classes=num_classes,
            no_of_inputs=no_of_inputs,
            name="BayesianCNN",
        )

# Define input shape, number of classes, KL divergence function, and task
input_shape = (1121, 1)  # Example input shape
num_classes = 10  # Example number of classes
kl_divergence_fn = None  # Example KL divergence function (you need to define this)
task = "regression"  # Example task

# Instantiate the BayesianCNN model with all required arguments
bayesian_cnn = BayesianCNN(input_shape, num_classes, kl_divergence_fn, task)

# Plot the model
plot_model(bayesian_cnn, to_file='bayesian_cnn_model.png', show_shapes=True)
