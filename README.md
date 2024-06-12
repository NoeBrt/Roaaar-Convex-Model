---
# Fashion MNIST Model Optimization

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training and Testing](#training-and-testing)
  - [Results Visualization](#results-visualization)
- [Model and Testing Code](#model-and-testing-code)
- [References](#references)

## Project Overview
This project aims to optimize a Convolutional Neural Network (CNN) model for the Fashion MNIST dataset using various optimizers and learning rates. The goal is to find the best learning rate for different epoch configurations and visualize the results in a 3D graph.

## Directory Structure
```
.
├── fashion_mnist_model.py
├── test_fashion_mnist_model.py
├── results/
├── models/
└── README.md
```

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy
- Pandas
- NVIDIA CUDA
- NVIDIA cuDNN

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

Ensure that you have the necessary NVIDIA drivers and libraries installed on your system. You can download and install CUDA and cuDNN from the official NVIDIA website:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN SDK](https://developer.nvidia.com/cudnn)

## Usage

### Training and Testing
1. **Load and Preprocess Data**: Load the Fashion MNIST dataset and preprocess it for training.
2. **Build and Train Model**: Build a CNN model and train it using different optimizers and learning rates.
3. **Evaluate Model**: Evaluate the model to find the best learning rate for each epoch configuration.

To run the tests and generate the results, execute the `test_fashion_mnist_model.py` script:
```bash
python test_fashion_mnist_model.py
```

### Results Visualization
The results will be saved in the `results/` directory as 3D plots. Each plot's title will include the optimizer name and epoch number, showing the relationship between learning rate, validation accuracy, and epochs.

## Model and Testing Code

### FashionMNISTModel Class
The `FashionMNISTModel` class in `fashion_mnist_model.py` handles data loading, preprocessing, model building, training, and evaluation.

### TestFashionMNISTModel Class
The `TestFashionMNISTModel` class in `test_fashion_mnist_model.py` tests the model with various learning rates and epochs, and visualizes the results.

## References
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Lion Optimizer Article](https://medium.com/@yash9439/lion-optimizer-73d3fd18abe9)

*"le lion rugit, s'elance dans la savane" Angèle*
---
