# Breast Cancer Classification using Artificial Neural Network (ANN)

## Overview

This project implements a Breast Cancer Classification model using an Artificial Neural Network (ANN) with Scikit-learn's MLPClassifier.

The model uses the Breast Cancer Wisconsin dataset to classify tumors into two categories:

- Malignant: Cancerous tumor.
- Benign: Non-cancerous tumor.

The objective is to understand how a Multi-Layer Perceptron (MLP) learns nonlinear patterns from numerical features and performs binary classification.

## Dataset

The project uses the built-in Breast Cancer Wisconsin dataset available through sklearn.datasets.

- Total Samples: 569
- Total Features: 30
- Target Classes: 2
- Classification Type: Binary Classification

The dataset contains numerical measurements describing characteristics of cell nuclei.

## Model Architecture

The project uses a Multi-Layer Perceptron (MLP) neural network.

| Parameter | Value |
|---|---|
| Model | MLPClassifier |
| Input Features | 30 |
| Hidden Layers | 2 |
| Hidden Neurons | (16, 8) |
| Activation Function | ReLU |
| Optimizer | Adam |
| Maximum Iterations | 1000 |
| Random State | 42 |

## Implementation Workflow

1. Load the Breast Cancer Wisconsin dataset.
2. Extract input features and target labels.
3. Explore dataset dimensions and class information.
4. Split the dataset into training and testing sets (80:20).
5. Apply StandardScaler for feature scaling.
6. Build the ANN model using MLPClassifier.
7. Train the model using the training dataset.
8. Generate predictions on the testing dataset.
9. Evaluate model performance using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
   - Precision
   - Recall
   - F1-Score
10. Test the model with a sample input and examine prediction probabilities.

## Technologies Used

- Python
- NumPy
- Scikit-learn
- Artificial Neural Networks
- Multi-Layer Perceptron (MLP)
- StandardScaler

## Project Structure

```text
Breast-Cancer-Classification-using-ANN/
│
├── ANN_MLP.py
├── requirements.txt
└── README.md
