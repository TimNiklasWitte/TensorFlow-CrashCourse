from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

class Classifier(tf.keras.Model):

  def __init__(self):
    super(Classifier, self).__init__()

    self.layer_list = [
      #tf.keras.layers.Dense(32, activation="tanh"),
      tf.keras.layers.Dense(16, activation="tanh"),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    self.loss_function = tf.keras.losses.BinaryCrossentropy()

    self.metric_loss = tf.keras.metrics.Mean(name="loss")
    self.metric_accuracy = tf.keras.metrics.Accuracy(name="accuracy")


  @tf.function
  def call(self, x):
    for layer in self.layer_list:
      x = layer(x)
    return x

  @tf.function
  def train_step(self, image, target):

    with tf.GradientTape() as tape:
      prediction = self(image)
      loss = self.loss_function(target, prediction)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    self.metric_loss.update_state(loss)

    prediction = tf.argmax(prediction, axis=-1)
    label = tf.argmax(target, axis=-1)
    self.metric_accuracy.update_state(label, prediction)

  def test_step(self, dataset):
          
    self.metric_loss.reset_states()
    self.metric_accuracy.reset_states()

    for image, target in dataset:
        prediction = self(image)

        loss = self.loss_function(target, prediction)
        self.metric_loss.update_state(loss)

        prediction = tf.argmax(prediction, axis=-1)
        label = tf.argmax(target, axis=-1)
        self.metric_accuracy.update_state(label, prediction)
