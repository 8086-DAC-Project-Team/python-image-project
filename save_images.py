import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Loading MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Selecting a specific image - for example image number 3
index = 7
original_image = x_train[index]

# Applying Thresholding filter
def apply_threshold(img, threshold=100):
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

thresholded_image = apply_threshold(original_image)

# Displaying both images side by side
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("After Thresholding")
plt.imshow(thresholded_image, cmap='gray')
plt.axis('off')

plt.show()

# Saving images to disk
cv2.imwrite("original_image.png", original_image)
cv2.imwrite("thresholded_image.png", thresholded_image)

print("✔️ Images saved: original_image.png and thresholded_image.png")
