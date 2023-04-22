import tensorflow as tf

class Decoder(tf.keras.Model): # <-- Needed to make parameters trainable and to be callable
    def __init__(self):

        super(Decoder, self).__init__()
        self.layer_list = [
            
            tf.keras.layers.Conv2DTranspose(105, kernel_size=(3,3), strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),

            tf.keras.layers.Conv2DTranspose(90, kernel_size=(3,3), strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),

            tf.keras.layers.Conv2DTranspose(75, kernel_size=(3,3), strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),

            # bottleneck to RGB

            tf.keras.layers.Conv2DTranspose(2, kernel_size=(1,1), strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.tanh),
            
        ]
    
    @tf.function
    def call(self, x, training):
        
        #print("decoder:")
        for layer in self.layer_list:
            #print(x.shape)
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x,training)
            else: 
                x = layer(x)
        # print(x.shape) 
        # print("-------------")
        # exit()
        return x