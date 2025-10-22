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

<img width="885" height="1030" alt="image" src="https://github.com/user-attachments/assets/7eb04427-1e64-4867-8d84-95859d819c99" />

---

## 5. Visualizations

### Model Performance

Training and validation accuracy and loss curves across 10 epochs. The model shows rapid convergence with minimal overfitting, achieving high accuracy (~99.7% training and ~98.9% validation), indicating successful learning

<img width="1600" height="857" alt="image" src="https://github.com/user-attachments/assets/e2d5f14e-d4e6-4367-b003-df745d81855d" />
<img width="1044" height="1030" alt="image" src="https://github.com/user-attachments/assets/390ff378-600f-405f-93b5-4939c0edc166" />


---

### Sample Digit Grid

A grid of 25 sample images from MNIST, showing handwritten digits (0–9) with their labels to visualize and verify dataset correctness.

<img width="1600" height="860" alt="image" src="https://github.com/user-attachments/assets/fb807161-0965-4d2c-8061-809bbe2c92f3" />
<img width="819" height="1030" alt="image" src="https://github.com/user-attachments/assets/11b8d4f2-4c8b-4f04-9110-03c46ecac3df" />


---

### Thresholding Example

The original and thresholded versions of a handwritten digit "3". Thresholding filter was applied to enhance the contrast and simplify the image, converting it into a binary format (black and white) for better feature extraction.

<img width="1600" height="860" alt="image" src="https://github.com/user-attachments/assets/d96aef13-e1b5-44ef-9249-f3389185e269" />

---

## 6. Conclusion

Applying the Thresholding filter successfully simplified the input images while preserving essential classification features.  
As a result, the CNN model achieved high accuracy, demonstrating the effectiveness of preprocessing (thresholding) in digit classification tasks.

