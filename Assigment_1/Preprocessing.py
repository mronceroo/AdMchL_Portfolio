import tensorflow as tf
from keras import datasets, layers, models, optimizers, losses
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = (train_images.astype('float32') / 255) * 2 - 1
test_images = (test_images.astype('float32') / 255) * 2 - 1

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Show some training images
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(1, 4, i+1)
    img = (train_images[i] + 1) / 2  # Scale back to [0,1] for display
    plt.imshow(img)
    plt.title(class_names[train_labels[i][0]])
    plt.axis('off')
plt.show()