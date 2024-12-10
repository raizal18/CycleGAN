import tensorflow as tf
import os

def load_dataset(dataset_path, image_size=(256, 256), batch_size=32):
    """
    Loads the datasets from the given directory, processes them, and returns TensorFlow datasets
    for training and testing.

    Args:
        dataset_path (str): Path to the dataset folder (selfie2anime).
        image_size (tuple): Desired image size to resize the images (height, width).
        batch_size (int): Number of images per batch.

    Returns:
        train_dataset, test_dataset: TensorFlow Datasets for training and testing.
    """
    
    trainA_dir = os.path.join(dataset_path, 'trainA')
    trainB_dir = os.path.join(dataset_path, 'trainB')
    testA_dir = os.path.join(dataset_path, 'testA')
    testB_dir = os.path.join(dataset_path, 'testB')

    trainA_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        trainA_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=None,
        shuffle=True
    )

    trainB_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        trainB_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=None,
        shuffle=True
    )

    testA_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        testA_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=None,
        shuffle=False  
    )

    testB_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        testB_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode=None,
        shuffle=False
    )
    
    train_dataset = tf.data.Dataset.zip((trainA_dataset, trainB_dataset))
    test_dataset = tf.data.Dataset.zip((testA_dataset, testB_dataset))

    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset