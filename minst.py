import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Loading MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Displaying the first 25 images as examples
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.title(f"Label: {y_train[i]}")

plt.show()
