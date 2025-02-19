import cv2
import numpy as np
import tensorflow as tf

# Loas model and prepare Class names
model = tf.keras.models.load_model("fashion_mnist.h5")

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Camera settings
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    # Convert gary scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))

    # Normalize img
    img = (resized.astype('float32') / 255) * 2 - 1
    img = img.reshape(1, 28, 28, 1)  
    
    # Aplyy model to do predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    class_label = class_names[predicted_class]