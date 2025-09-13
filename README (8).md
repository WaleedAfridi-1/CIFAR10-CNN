# MNIST Handwritten Digit Classification (ANN - Keras)

This project demonstrates how to build an **Artificial Neural Network (ANN)** using **Keras (TensorFlow backend)** to classify handwritten digits from the famous [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

---

## 📌 Project Overview
- Dataset: **MNIST** (70,000 grayscale images of handwritten digits 0–9)
- Image size: **28x28 pixels**
- Classes: **10 (digits 0–9)**
- Model: **ANN (Multi-Layer Perceptron)** built with **Keras Sequential API**
- Goal: Train a neural network to correctly classify handwritten digits.

---

## 📂 Dataset
- **Training Data:** 60,000 images  
- **Test Data:** 10,000 images  
- Images are normalized (0–255 → 0–1) for faster convergence.

---

## 🏗️ Model Architecture
```python
model = Sequential([
    Flatten(input_shape=(28, 28)),       # Flatten 28x28 → 784
    Dense(32, activation='relu'),        # Hidden Layer 1
    Dense(16, activation='relu'),        # Hidden Layer 2
    Dense(8, activation='relu'),         # Hidden Layer 3
    Dense(10, activation='softmax')      # Output Layer (10 classes)
])
```

- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Metrics:** Accuracy  

---

## 📊 Training Results
- Training epochs: **5–10**
- Final Accuracy: ~ **97% on validation set**
- Loss decreases smoothly, showing good learning.

---

## 📈 Visualization
The following plots are included in the notebook:

1. **Accuracy Plot**  
   - Training vs Validation Accuracy  
2. **Loss Plot**  
   - Training vs Validation Loss  
3. **Confusion Matrix**  
   - Shows class-wise performance  

---

## 🔧 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mnist-ann.git
   cd mnist-ann
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook MNIST_ANN.ipynb
   ```

---

## 📜 Requirements
- Python 3.x  
- TensorFlow / Keras  
- Matplotlib  
- Scikit-learn  

Install via:
```bash
pip install tensorflow matplotlib scikit-learn
```

---

## 📌 Output Examples

### Accuracy vs Validation Accuracy
![Accuracy Plot](images/accuracy.png)

### Loss vs Validation Loss
![Loss Plot](images/loss.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

---

## 🤝 Contributing
Feel free to fork this repo and improve the model (e.g., add Dropout, BatchNormalization, CNN layers).

---

## 📧 Contact
Developed by **Waleed Afridi**  
From: **Kohat, KPK**  
