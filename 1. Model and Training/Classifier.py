"""
Layers
Optimizer
Loss function
Metrics
train and test step
"""

import tensorflow as tf
from MyDenseLayer import *
from MyResidualLayer import *

class Classifier(tf.keras.Model):

    def __init__(self):
        """
        Create the Classifier.
        """

        super(Classifier, self).__init__()

        self.layer_list = [
            # tf.keras.layers.Dense(32, activation="tanh"),
            # MyDenseLayer(20, tf.nn.tanh),
            # MyResidualLayer(10), 
            tf.keras.layers.Dense(16, activation="tanh"),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")


    @tf.function
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

    @tf.function
    def train_step(self, image, target):
        """
        Train this model. This process updates the loss and accuracy metric.

        Args:
            image -- batch of flatten images, shape: (batch_size, 784)
            target -- batch of target predictions (one-hot vectors) -> shape: (10, 784)
        """

        with tf.GradientTape() as tape:
            prediction = self(image)
            loss = self.loss_function(target, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        prediction = tf.argmax(prediction, axis=-1)
        label = tf.argmax(target, axis=-1)
        self.metric_accuracy.update_state(label, prediction)

    @tf.function
    def test_step(self, dataset):
        """
        Creates a prediction of each batch of images in the dataset.
        Now the loss and accuracy metrics are updated.

        Args:
            dataset -- containing batches of images.
        """

        self.metric_loss.reset_states()
        self.metric_accuracy.reset_states()

        for image, target in dataset:
            prediction = self(image)

            loss = self.loss_function(target, prediction)
            self.metric_loss.update_state(loss)

            prediction = tf.argmax(prediction, axis=-1)
            label = tf.argmax(target, axis=-1)
            self.metric_accuracy.update_state(label, prediction)
