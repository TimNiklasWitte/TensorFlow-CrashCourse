import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, units, activation):
        super(MyDenseLayer, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                        initializer='random_normal',
                        trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                        initializer='random_normal',
                        trainable=True)

    @tf.function 
    def call(self, inputs):
        
        drive = tf.matmul(inputs, self.w) + self.b 
        y = self.activation(drive)
        return y   