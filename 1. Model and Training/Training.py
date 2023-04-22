import tensorflow as tf
import numpy as np

import os
import tqdm

from Classifier import *

def train_ds_generator():
    path = "./dataset/train_data/"
    files = os.listdir(path)
    for fileName in files:
        data = np.loadtxt(path + fileName, dtype=np.uint8)

        label = fileName.split("_")[3][:-4]

        if label == '':
            continue

        label = int(label)
        yield data, label

def test_ds_generator():
    path = "./dataset/test_data/"
    files = os.listdir(path)
    for fileName in files:
        data = np.loadtxt(path + fileName, dtype=np.uint8)

        label = fileName.split("_")[3][:-4]

        if label == '':
            continue

        label = int(label)
        yield data, label

def prepare_digit_data(digit):
    #flatten the images into vectors
    digit = digit.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    #create one-hot targets
    digit = digit.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    digit = digit.cache()
    #shuffle, batch, prefetch

    #mnist = mnist.map(lambda img, target: (tf.math.ceil(img), target))

    digit = digit.shuffle(1350)
    digit = digit.batch(32)
    digit = digit.prefetch(20)
    #return preprocessed dataset
    return digit


def main():
          
    train_ds = tf.data.Dataset.from_generator(
                    train_ds_generator, 
                    output_signature=(
                            tf.TensorSpec(shape=(28,28), dtype=tf.uint8),
                            tf.TensorSpec(shape=(), dtype=tf.uint8)
                        )
                )

    test_ds = tf.data.Dataset.from_generator(
                    test_ds_generator, 
                    output_signature=(
                            tf.TensorSpec(shape=(28,28), dtype=tf.uint8),
                            tf.TensorSpec(shape=(), dtype=tf.uint8)
                        )
                )

    train_dataset = train_ds.apply(prepare_digit_data)
    test_dataset = test_ds.apply(prepare_digit_data)


    # Logging
    file_path = "test_logs/test" 
    train_summary_writer = tf.summary.create_file_writer(file_path)

    num_epochs = 50
    
    # Initialize the model.
    classifier = Classifier()

    log(train_summary_writer, classifier, train_dataset, test_dataset, 0)

    classifier.summary()

    #
    # Train loop
    #
    train_size = 1726
    batch_size = 32
    for epoch in range(num_epochs):
            
        print(f"Epoch {epoch}")

        for image, label in tqdm.tqdm(train_dataset, total=int(train_size/batch_size)): 
            classifier.train_step(image, label)

        log(train_summary_writer, classifier, train_dataset, test_dataset, epoch + 1)
    
        classifier.save_weights(f"./saved_models/trained_weights_{epoch + 1}", save_format="tf")

def log(train_summary_writer, classifier, train_dataset, test_dataset, epoch):

    with train_summary_writer.as_default():
        
        if epoch == 0:
            classifier.test_step(train_dataset.take(1000))

   
        loss = classifier.metric_loss.result()
        tf.summary.scalar(f"train_loss", loss, step=epoch)
        
        accuracy = classifier.metric_accuracy.result()
        tf.summary.scalar(f"train_accuracy", accuracy, step=epoch)

        print(f"train_loss: {loss}")
        print(f"train_accuracy: {accuracy}")

        classifier.metric_loss.reset_states()
        classifier.metric_accuracy.reset_states()

        classifier.test_step(test_dataset)

        loss = classifier.metric_loss.result()
        tf.summary.scalar(f"test_loss", loss, step=epoch)

        accuracy = classifier.metric_accuracy.result()
        tf.summary.scalar(f"test_accuracy", accuracy, step=epoch)

        print(f"test_loss: {loss}")
        print(f"test_accuracy: {accuracy}")

        classifier.metric_loss.reset_states()
        classifier.metric_accuracy.reset_states()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")