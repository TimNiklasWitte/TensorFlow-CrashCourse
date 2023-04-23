import tensorflow as tf
from Classifier import *

def main():
    inputs = tf.keras.Input(shape=(32,784), name="Flatten image") 

    classifier = Classifier()
    classifier = tf.keras.Model(inputs=[inputs],outputs=classifier.call(inputs))

    tf.keras.utils.plot_model(classifier,show_shapes=True, show_layer_names=True, to_file="ClassifierPlot.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")