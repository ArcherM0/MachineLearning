""" Practice Keras machine learning model using Keras data sets """


""" Fashion_mnist dataset example (images classification) """
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
(train_images, train_labels), (test_images, test_label) = fashion_mnist.load_data()

label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


datagen = ImageDataGenerator(
    rescale = 1./255
    featurewise_center = True,
    featurewise_std_normalization = True,
    rotation_range = 10,
    horizontal_flip = True)


datagen.fit(train_images)

test_images = teat_images / 255.0

model = keras.sequential([
  keras.layers.Flatten(input_shape = (28,28)),
  keras.layers.Dense(128, activation = tf.nn.relu),
  keras.layers.Dense(10, activation = tf.nn.softmax)
])








