"""
Layers
Optimizer
Loss function
Metrics
train and test step
"""

import tensorflow as tf

class Classifier(tf.keras.Model):

    def __init__(self):
        """
        Create the Classifier.
        """

        super(Classifier, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(16, activation="tanh"),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")

    # REMOVE @tf.function !!!!!
    #@tf.function 
    def call(self, x):
        """
        Pass the input x of shape (batch_size, 784) (batch of flatten images) 
        through each layer and returns a predication (batch_size, 10)

        Args:
            x -- input tensor of shape (batch_size, 784)
        
        Return:
            Prediction tensor (batch_size, 10)
        """

        for layer in self.layer_list:
            x = layer(x)
        return x

