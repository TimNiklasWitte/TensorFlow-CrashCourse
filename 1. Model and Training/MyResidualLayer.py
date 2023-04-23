"""
Own implementation of a residual layer
"""

import tensorflow as tf

class MyResidualLayer(tf.keras.layers.Layer):

    def __init__(self, units):
        """
        Create a residual layer.

        Args:
            units -- number of neurons
        """

        super(MyResidualLayer, self).__init__()
        self.layer_1 = tf.keras.layers.Dense(units)
        self.layer_2 = tf.keras.layers.Dense(units)

    def build(self, input_shape):
        """
        The build method on the two internal layers is called 
        with 'input_shape' as a argument.

        Args:
            input_shape -- Shape of the input
        """

        self.layer_1.build(input_shape)
        self.layer_2.build(input_shape)

    @tf.function 
    def call(self, x):
        """
        Determinate the output of the layer based on the input x.

        Args:
            x -- Tensor of the shape of 'input_shape' (see build)

        Returns:
            output of this layer
        """
        
        x_old = self.layer_1(x)
        x = self.layer_2(x)

        return x + x_old 