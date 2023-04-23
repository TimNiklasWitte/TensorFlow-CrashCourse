"""
Loading of the data
Preprocessing it
Training loop of the model + logging + saving weights per epoch
"""

import tensorflow as tf
import numpy as np

import os
import tqdm

from Classifier import *

NUM_EPOCHS = 50
BATCH_SIZE = 32

def dataset_generator(use_train: bool):
    """
    Yields a data point (image) and 
    its corresponding label of the specified dataset 
    by use_train.

    Args:
        use_train: boolean, indicate wheater data is 
                   loaded from the train or test dataset
                      true -> train dataset
                      false -> test dataset
    """
    
    # Can not pass strings -> Use boolean instead
    if use_train:
        path = "./dataset/train_data/"
    else:
        path = "./dataset/test_data/"

    files = os.listdir(path)

    for fileName in files:
        data = np.loadtxt(path + fileName, dtype=np.uint8)

        # DataNo_x_Label_y 
        # x = id of data point  
        # y = label 
        label = fileName.split("_")[3][:-4]

        if label == '':
            continue

        label = int(label)
        yield data, label

def main():
    
    #  
    # Create train and test dataset
    #

    train_ds = tf.data.Dataset.from_generator(
                    dataset_generator,
                    args=(True,),
                    output_signature=(
                            tf.TensorSpec(shape=(28,28), dtype=tf.uint8),
                            tf.TensorSpec(shape=(), dtype=tf.uint8)
                        )
                )

    test_ds = tf.data.Dataset.from_generator(
                    dataset_generator,
                    args=(False,),
                    output_signature=(
                            tf.TensorSpec(shape=(28,28), dtype=tf.uint8),
                            tf.TensorSpec(shape=(), dtype=tf.uint8)
                        )
                )

    # Preprocessing pipeline
    train_dataset = train_ds.apply(prepare_data)
    test_dataset = test_ds.apply(prepare_data)

    #
    # Logging
    #
    file_path = "test_logs/test" 
    train_summary_writer = tf.summary.create_file_writer(file_path)

    #
    # Initialize model
    #
    classifier = Classifier()

    log(train_summary_writer, classifier, train_dataset, test_dataset, 0)

    classifier.summary()

    #
    # Train loop
    #
    train_size = 1726 # hard coded :D
    for epoch in range(1, NUM_EPOCHS):
            
        print(f"Epoch {epoch}")

        for image, label in tqdm.tqdm(train_dataset, total=int(train_size/BATCH_SIZE), leave=False): 
            classifier.train_step(image, label)
     
        log(train_summary_writer, classifier, train_dataset, test_dataset, epoch)
    
        classifier.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")


def log(train_summary_writer, classifier, train_dataset, test_dataset, epoch):

    # Epoch 0 = no training steps are performed 
    # test based on train data
    # -> Determinate initial train_loss and train_accuracy
    if epoch == 0:
        classifier.test_step(train_dataset.take(5000))

    #
    # Train
    #
    train_loss = classifier.metric_loss.result()
    train_accuracy = classifier.metric_accuracy.result()

    classifier.metric_loss.reset_states()
    classifier.metric_accuracy.reset_states()

    #
    # Test
    #

    classifier.test_step(test_dataset)

    test_loss = classifier.metric_loss.result()
    test_accuracy = classifier.metric_accuracy.result()

    classifier.metric_loss.reset_states()
    classifier.metric_accuracy.reset_states()

    #
    # Write to TensorBoard
    #
    with train_summary_writer.as_default():
        tf.summary.scalar(f"train_loss", train_loss, step=epoch)
        tf.summary.scalar(f"train_accuracy", train_accuracy, step=epoch)

        tf.summary.scalar(f"test_loss", test_loss, step=epoch)
        tf.summary.scalar(f"test_accuracy", test_accuracy, step=epoch)

    #
    # Output
    #
    print(f"    train_loss: {train_loss}")
    print(f"     test_loss: {test_loss}")
    print(f"train_accuracy: {train_accuracy}")
    print(f" test_accuracy: {test_accuracy}")

def prepare_data(dataset):
    """
    Returns a preprocessed dataset by the following stages:
    1. Flatten
    2. Cast uint8 -> float32
    3. Create one-hot targets
    4. Caching
    5. Shuffle
    6. Batch
    7. Prefetch

    Args:
        dataset: TensorFlow dataset to preprocess

    Return:
        Preprocessed dataset
    """

    # Flatten images into vectors
    dataset = dataset.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    
    # Convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target) )

    # No input normalization needed
    # Data is already in range of [0, 1]

    # Create one-hot targets
    dataset = dataset.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    
    # Caching: Save already performed proprocessing steps
    dataset = dataset.cache()

    #shuffle, batch, prefetch
    dataset = dataset.shuffle(1350)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")