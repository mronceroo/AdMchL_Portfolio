import cv2
import numpy as np
import tensorflow as tf

# Loas model and prepare Class names
model = tf.keras.models.load_model("fashion_mnist.h5")

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]