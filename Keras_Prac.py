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

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.sequential([
  keras.layers.Flatten(input_shape = (28,28)),
  keras.layers.Dense(128, activation = tf.nn.relu),
  keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model.evaluate(test_images, test_labels)










