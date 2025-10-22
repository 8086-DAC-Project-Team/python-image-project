# pip install tensorflow opencv-python matplotlib

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
# 1. Loading MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 2. Applying Thresholding filter
def apply_threshold(img, threshold=100):
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

x_train_filtered = np.array([apply_threshold(img) for img in x_train])
x_test_filtered = np.array([apply_threshold(img) for img in x_test])
# Normalizing data
x_train_filtered = x_train_filtered / 255.0
x_test_filtered = x_test_filtered / 255.0
# Expanding the third dimension for CNN
x_train_filtered = np.expand_dims(x_train_filtered, axis=-1)
x_test_filtered = np.expand_dims(x_test_filtered, axis=-1)
# 3. Building CNN network
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
#adam = Adaptive Moment Estimation.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 4. Training the model
history = model.fit(x_train_filtered, y_train, epochs=10, batch_size=32,
                    validation_data=(x_test_filtered, y_test))
# 5. Plotting the results
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curve')

plt.show()
