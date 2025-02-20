import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, optimizers, losses
import numpy as np


model = keras.models.load_model("Assigment_1/Manuel_Roncero_CNN.h5")

# Freeze last layers
for layer in model.layers[:-1]:
    layer.trainable = False

# Modify last layer for MNIST 10 classes
model.pop()
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = (train_images.astype('float32') / 255) * 2 - 1
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = (test_images.astype('float32') / 255) * 2 - 1
test_images = test_images.reshape(-1, 28, 28, 1)

# Train MNIST
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

model.save("FineTuned_MNIST.h5")

# --- Transfer Learning to CIFAR-10 ---

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = (train_images.astype('float32') / 255) * 2 - 1
test_images = (test_images.astype('float32') / 255) * 2 - 1

# Modify first layer to acept img 32x32x3
model.layers[0] = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))

# Modify last layer for CIFAR-10 10 classes
model.pop()
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train in CIFAR-10
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))


model.save("FineTuned_CIFAR10.h5")
