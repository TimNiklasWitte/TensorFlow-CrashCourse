import tensorflow as tf
import numpy as np

from Encoder import * 
from Decoder import *

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.metric_mean = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def call(self, x, training=False):
        embedding = self.encoder(x, training)
        decoded = self.decoder(embedding, training)
        return decoded

    @tf.function
    def train_step(self, input, target):

        with tf.GradientTape() as tape:
            prediction = self(input, training=True)
            loss = self.loss_function(target, prediction)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_mean.update_state(loss)

        

    def test(self, test_data):

        self.metric_mean.reset_states()

        # test over complete test data
        for input, target in test_data:           
            prediction = self(input)
            
            loss = self.loss_function(target, prediction)
            self.metric_mean.update_state(loss)

        mean_loss = self.metric_mean.result()
        self.metric_mean.reset_states()
        return mean_loss

    
