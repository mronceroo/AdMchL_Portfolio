import cv2
import numpy as np
import tensorflow as tf

# Loas model and prepare Class names
model = tf.keras.models.load_model("Assigment_1/Manuel_Roncero_CNN.h5")

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
    
    #Show predic and image
    cv2.putText(frame, f"Prediction: {class_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow("Fashion MNIST Classifier", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()