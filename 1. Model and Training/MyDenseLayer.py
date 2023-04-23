"""
Own implementation of a dense layer
"""

import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, units, activation):
        """
        Create a dense layer .

        Args:
            units -- number of neurons
            activation -- activation function
        """

        super(MyDenseLayer, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        """
        Build the weight matrix and the bias vector (parameters of this layer).

        Args:
            input_shape -- Shape of the input
        """

        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                        initializer='random_normal',
                        trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                        initializer='random_normal',
                        trainable=True)

    @tf.function 
    def call(self, x):
        """
        Determinate the output of the layer based on the input x.

        Args:
            x -- Tensor of the shape of 'input_shape' (see build)

        Returns:
            output of this layer
        """
        
        drive = tf.matmul(x, self.w) + self.b 
        y = self.activation(drive)
        return y   