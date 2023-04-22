import tensorflow as tf

class Encoder(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer_list = [
            # input (243,243) 

            tf.keras.layers.Conv2D(75, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            # -> (81, 81, 32)   

            tf.keras.layers.Conv2D(90, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            # -> (27, 27, 64)

            tf.keras.layers.Conv2D(105, kernel_size=(3, 3), strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),


            # bottleneck
            tf.keras.layers.Conv2D(3, kernel_size=(1, 1), strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            
        ]
    
    @tf.function
    def call(self, x, training):
        #print("encoder:")
        for layer in self.layer_list: 
            #print(x.shape)
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x,training)
            else: 
                x = layer(x)
            
        #print(x.shape)
        #print("-------------")
        #exit()
        return x