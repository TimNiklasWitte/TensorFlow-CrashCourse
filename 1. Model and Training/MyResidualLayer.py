import tensorflow as tf

class MyResidualLayer(tf.keras.layers.Layer):

    def __init__(self, units):
        super(MyResidualLayer, self).__init__()
        self.layer_1 = tf.keras.layers.Dense(units)
        self.layer_2 = tf.keras.layers.Dense(units)


    @tf.function 
    def call(self, x):
        
        x_old = self.layer_1(x)
        x = self.layer_2(x)

        return x + x_old 