# MNIST Digit Classification with CNN

## 1. Dataset Description
We used the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

| Class | Number of Samples |
|-------|-------------------|
| 0–9   | ~6000-7000 images per class |

---

## 2. Applied Filter: Thresholding
We applied a Thresholding filter at value 100 to convert grayscale images to binary (black/white).

**Purpose:**
- Highlight digit structure
- Simplify image features for easier classification by the CNN model

---

## 3. CNN Architecture

| Layer | Type                       | Parameters             |
|-------|----------------------------|------------------------|
| 1     | Conv2D (32 filters, 3x3)   | Input: (28, 28, 1)     |
| 2     | MaxPooling2D (2x2)         |                        |
| 3     | Conv2D (64 filters, 3x3)   |                        |
| 4     | MaxPooling2D (2x2)         |                        |
| 5     | Flatten                    |                        |
| 6     | Dense (128 neurons, ReLU)  |                        |
| 7     | Dense (10 neurons, Softmax)|                        |

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Epochs:** 10
- **Batch Size:** 32

---

## 4. Results

- Training Accuracy: **~99.8%**
- Validation Accuracy: **~98.86%**

---

## 5. Visualizations

### Model Performance

Training and validation accuracy and loss curves across 10 epochs.  
The model converges rapidly with minimal overfitting.

<img width="885" height="1030" alt="image" src="https://github.com/user-attachments/assets/d74f4698-0b82-4341-82da-eb2e2c32f777" />

---

### Sample Digit Grid

A grid of 25 sample images from MNIST, showing handwritten digits (0–9) with their labels to visualize and verify dataset correctness.

![MNIST Grid](attached_image:5)

---

### Thresholding Example

Original and thresholded versions of a handwritten "3".
Thresholding enhances contrast, converting the image into binary for feature extraction.

![Thresholding Example](attached_image:4)

---

## 6. Conclusion

Applying the Thresholding filter successfully simplified the input images while preserving essential classification features.  
As a result, the CNN model achieved high accuracy, demonstrating the effectiveness of preprocessing (thresholding) in digit classification tasks.

