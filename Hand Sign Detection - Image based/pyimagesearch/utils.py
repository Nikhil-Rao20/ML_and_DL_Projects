from pyimagesearch import config
import tensorflow as tf

def prepare_batch_dataset(data_path, img_size, batch_size, shuffle=True):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        image_size=(img_size, img_size),
        shuffle=shuffle,
        batch_size=batch_size
    )

def callbacks():
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=2,
            mode="auto",
        ),
    ]
    return callbacks

def normalize_layer(factor = 1./127.5):
    return tf.keras.layers.Rescaling(factor, offset=-1)

def augmentation():
    data_aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1)
        ]
    )
    return data_aug
