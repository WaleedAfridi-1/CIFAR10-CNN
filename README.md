# CIFAR10-CNN
This project implements an image classification model on the CIFAR-10 dataset using CNN with Batch Normalization and Dropout. The model achieves up to 87% accuracy with techniques like data augmentation, learning rate scheduling, and early stopping for better generalization.


# CIFAR-10 Image Classification using CNN

This project implements an image classification model on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) with Batch Normalization, Dropout, and Data Augmentation. The model achieves up to **87% accuracy** using learning rate scheduling and early stopping.

---

## 📌 Short Description
This project implements an image classification model on the CIFAR-10 dataset using CNN with Batch Normalization and Dropout. The model achieves up to 87% accuracy with techniques like data augmentation, learning rate scheduling, and early stopping for better generalization.

---

## 📊 Dataset
- **CIFAR-10** dataset (60,000 images, 32x32x3 size, 10 classes).  
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

---

## 🛠️ Technologies Used
- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- scikit-learn  

---

## 🚀 Model Architecture
- **Conv Block 1**: Conv2D (32 filters) → BatchNorm → Conv2D → BatchNorm → MaxPooling → Dropout  
- **Conv Block 2**: Conv2D (64 filters) → BatchNorm → Conv2D → BatchNorm → MaxPooling → Dropout  
- **Conv Block 3**: Conv2D (128 filters) → BatchNorm → Conv2D → BatchNorm → MaxPooling → Dropout  
- **Flatten Layer**  
- **Dense Layer**: 512 neurons + ReLU + Dropout  
- **Output Layer**: 10 neurons (Softmax)  

---

## ⚙️ Training Setup
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Metrics: Accuracy  
- Callbacks:  
  - ReduceLROnPlateau  
  - EarlyStopping  

---

## 📈 Results
- Training Accuracy: ~87%  
- Validation Accuracy: ~87%  
- Loss reduced significantly with BatchNorm + Dropout + Augmentation.  

---

## 📉 Visualizations
- Accuracy Curve (Train vs Validation)  
- Loss Curve (Train vs Validation)  
- Confusion Matrix for class-wise performance  

---

## 📂 How to Run
```bash
git clone <your-repo-link>
cd cifar10-cnn
pip install -r requirements.txt
python train.py
```

---

## ✨ Future Improvements
- Use ResNet / EfficientNet for higher accuracy  
- Hyperparameter tuning with Keras Tuner  
- Deploy model with FastAPI / Streamlit  

---

## 🙌 Author
Developed by **Waleed Afridi**  
