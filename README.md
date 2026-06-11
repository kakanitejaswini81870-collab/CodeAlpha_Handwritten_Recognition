# Task: Task 3 - Handwritten Character Recognition
## CodeAlpha Machine Learning Internship
### 📌 Objective
To identify and classify handwritten digits (0-9) using Convolutional Neural Networks (CNN) on the MNIST dataset. This project demonstrates image processing and deep learning for Optical Character Recognition (OCR).
### 🎯 Approach
1. **Dataset**: Used MNIST dataset containing 60,000 training images and 10,000 test images of handwritten digits
2. **Preprocessing**: Normalized pixel values to range [0,1] and reshaped images to (28, 28, 1) format
3. **Model Architecture**: Built CNN using TensorFlow/Keras with Conv2D, MaxPooling, Flatten, Dense, and Dropout layers
4. **Training**: Used Adam optimizer with Sparse Categorical Crossentropy loss function for 5 epochs
5. **Evaluation**: Measured model performance using accuracy metrics and test validation
### ✨ Key Features
- Image preprocessing and normalization pipeline
- CNN architecture with 2x Conv2D + MaxPooling layers for feature extraction
- Dropout layer (0.5) to prevent overfitting
- Achieved 98%+ test accuracy on unseen data
- Model saved as `.h5` file for future predictions
- Training/validation accuracy visualization
### 🛠️ Tech Stack
- **Language**: Python 3.x
- **Libraries**: TensorFlow 2.x, Keras, NumPy, Matplotlib
- **Dataset**: MNIST (Inbuilt in Keras)
### 📊 Results
- **Test Accuracy**: 98.5%+
- **Training Epochs**: 5
- **Batch Size**: 128
- **Optimizer**: Adam
- **Model Type**: Sequential CNN
### 🚀 How to Run
```bash
pip install -r requirements.txt
python handwritten_digit.py
