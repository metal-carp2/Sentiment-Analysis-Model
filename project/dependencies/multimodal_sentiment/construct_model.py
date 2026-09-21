"""Single model definition for the supplied 40-MFCC feature dataset."""
import tensorflow as tf


def build_model(mean, variance, classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40,), name='mean_mfcc'),
        tf.keras.layers.Normalization(mean=mean, variance=variance, name='training_normalization'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(classes, activation='softmax', name='emotion'),
    ])
