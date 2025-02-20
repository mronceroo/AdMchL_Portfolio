import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, optimizers, losses
import numpy as np


model = keras.models.load_model("Assigment_1/Manuel_Roncero_CNN.h5")

# Freeze last layers
for layer in model.layers[:-1]:
    layer.trainable = False

# New model without the last layer
model_mnist = models.Sequential(model.layers[:-1])  # Copia todas menos la última
model_mnist.add(layers.Dense(10, activation='softmax', name="dense_mnist"))  # Nueva capa para MNIST


model_mnist.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = (train_images.astype('float32') / 255) * 2 - 1
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = (test_images.astype('float32') / 255) * 2 - 1
test_images = test_images.reshape(-1, 28, 28, 1)

# Train MNIST
model_mnist.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

model_mnist.save("FineTuned_MNIST.h5")

# --- Transfer Learning to CIFAR-10 ---

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = (train_images.astype('float32') / 255) * 2 - 1
test_images = (test_images.astype('float32') / 255) * 2 - 1

#New model for CIFAR-10
model_cifar10 = models.Sequential()

# Modify first layer to acept img 32x32x3
model_cifar10.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# Copy middle layers of the pprevious model
for layer in model_mnist.layers[1:-1]:  
    model_cifar10.add(layer)

# Add new exit layer
model_cifar10.add(layers.Dense(10, activation='softmax', name="dense_cifar10"))

model_cifar10.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

# Train in CIFAR-10
model_cifar10.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

model_cifar10.save("FineTuned_CIFAR10.h5")
