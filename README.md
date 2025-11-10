# Sign Language Recognition System

A deep learning-based sign language recognition system that uses Convolutional Neural Networks (CNN) to classify American Sign Language (ASL) letters from images. The model achieves **99.71% accuracy** on the validation set.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a CNN-based sign language recognition system that can classify 24 ASL letters (A-Z excluding J and Z, as they require motion). The system processes 28x28 grayscale images and uses a deep learning model to predict the corresponding sign language letter.

## ✨ Features

- **High Accuracy**: Achieves 99.71% accuracy on validation data
- **CNN Architecture**: Uses convolutional layers for feature extraction
- **Data Visualization**: Includes visualization of sample training images
- **Preprocessing**: Automatic data normalization and label encoding
- **Model Evaluation**: Comprehensive evaluation metrics

## 🏗️ Model Architecture

The model uses a sequential CNN architecture with the following layers:

1. **Conv2D Layer 1**: 32 filters, 3×3 kernel, ReLU activation
2. **MaxPooling2D**: 2×2 pooling
3. **Conv2D Layer 2**: 64 filters, 3×3 kernel, ReLU activation
4. **MaxPooling2D**: 2×2 pooling
5. **Flatten**: Converts 2D features to 1D
6. **BatchNormalization**: Normalizes activations
7. **Dense Layer**: 256 neurons with ReLU activation
8. **Dropout**: 30% dropout for regularization
9. **BatchNormalization**: Additional normalization
10. **Output Layer**: 24 neurons with softmax activation

**Total Parameters**: 442,264 (1.69 MB)

## 📊 Dataset

The model is trained on the **Sign Language MNIST** dataset, which contains:
- **Training samples**: 21,964 images
- **Validation samples**: 5,491 images
- **Image size**: 28×28 pixels (grayscale)
- **Classes**: 24 ASL letters (A-Z excluding J and Z)

The dataset is stored in CSV format where:
- First column contains the label (0-24)
- Remaining 784 columns contain pixel values (0-255)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/zortojnr/Sign-Language-Recognition-System.git
   cd Sign-Language-Recognition-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Place the `sign_mnist_train.csv` file in the `content/` directory
   - Optionally, add `sign_mnist_test.csv` for separate test evaluation

## 💻 Usage

### Running the Training Script

Simply run the main script:

```bash
python main.py
```

This will:
1. Load and preprocess the training data
2. Split data into training and validation sets
3. Display sample images with their labels
4. Build and compile the CNN model
5. Train the model for 10 epochs
6. Evaluate the model and display accuracy

### Expected Output

```
Training data shape: (21964, 28, 28, 1), Training labels shape: (21964, 24)
Validation data shape: (5491, 28, 28, 1), Validation labels shape: (5491, 24)

Model: "sequential"
...
[Model architecture summary]

Epoch 1/10
687/687 ━━━━━━━━━━━━━━━━━━━━ ... - accuracy: 0.9250 - loss: 0.2917 - val_accuracy: 0.9998 - val_loss: 0.0712
...
Epoch 10/10
687/687 ━━━━━━━━━━━━━━━━━━━━ ... - accuracy: 0.9978 - loss: 0.0087 - val_accuracy: 0.9971 - val_loss: 0.0091

Test Accuracy: 99.71%
```

## 📈 Results

### Training Performance

- **Final Training Accuracy**: 99.78%
- **Final Validation Accuracy**: 99.71%
- **Training Loss**: 0.0087
- **Validation Loss**: 0.0091

The model shows excellent generalization with minimal overfitting, achieving near-perfect accuracy on both training and validation sets.

## 📁 Project Structure

```
Sign-Language-Recognition-System/
│
├── main.py                 # Main training script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
│
└── content/
    └── sign_mnist_train.csv   # Training dataset
```

## 📦 Requirements

- **tensorflow** >= 2.13.0
- **numpy** >= 1.24.0
- **pandas** >= 2.0.0
- **matplotlib** >= 3.7.0
- **scikit-learn** >= 1.3.0

## 🔧 Code Structure

### Key Functions

- `load_data(path)`: Loads and preprocesses CSV data into image arrays and one-hot encoded labels
- Model compilation with Adam optimizer and categorical crossentropy loss
- Training with validation monitoring
- Model evaluation on test/validation data

### Data Preprocessing

- Reshapes pixel data from 784 columns to 28×28 images
- Normalizes pixel values to [0, 1] range
- Converts labels to one-hot encoded format
- Handles label adjustment (removes J and Z classes)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Zorto Jnr**
- GitHub: [@zortojnr](https://github.com/zortojnr)

## 🙏 Acknowledgments

- Sign Language MNIST dataset
- TensorFlow and Keras for deep learning framework
- The open-source community for tools and libraries

---

⭐ If you find this project helpful, please consider giving it a star!
