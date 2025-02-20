import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, optimizers, losses
import numpy as np


model = keras.models.load_model("Assigment_1/Manuel_Roncero_CNN.h5")

# Freeze last layers
for layer in model.layers[:-1]:
    layer.trainable = False

# New model without the last layer
model_mnist = models.Sequential(model.layers[:-1])  
model_mnist.add(layers.Dense(10, activation='softmax', name="dense_mnist"))  


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

#Add cnv layers form 0 for CIFAR-10
model_cifar10.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_cifar10.add(layers.MaxPooling2D((2, 2)))
model_cifar10.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cifar10.add(layers.MaxPooling2D((2, 2)))
model_cifar10.add(layers.Conv2D(128, (3, 3), activation='relu'))

model_cifar10.add(layers.GlobalAveragePooling2D())
model_cifar10.add(layers.Dense(128, activation='relu'))

# Copy middle layers of the pprevious model
for layer in model_mnist.layers[-2:]:  
    model_cifar10.add(layer)


model_cifar10.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss=losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

# Train in CIFAR-10
model_cifar10.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

model_cifar10.save("FineTuned_CIFAR10.h5")
