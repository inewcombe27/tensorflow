"""/Users/itn/github/tensorflow ."""
""" TensorFlow and tf.keras. """
import tensorflow as tf
import numpy as np

from tensorflow import keras

""" Helper libraries """

import matplotlib.pyplot as plt

print(tf.__version__)


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
