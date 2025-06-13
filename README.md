# MULTILAYER-NEURAL-NETWORK-
Breast Cancer Classification with Neural Network
This project implements a simple feedforward neural network with one hidden layer to classify breast tumors as benign or malignant using the Breast Cancer Wisconsin dataset.

Features
Loads and normalizes the dataset

Uses tanh activation in the hidden layer and sigmoid in the output

Computes binary cross-entropy loss

Trains with backpropagation and gradient descent

Evaluates accuracy on test data

Plots training loss curve over epochs

Requirements
Python 3.x

numpy

scikit-learn

matplotlib

Install dependencies with:

bash
Copy
Edit
pip install numpy scikit-learn matplotlib
Usage
Run the Python script to train the network and visualize the loss curve. The model typically achieves around 96% test accuracy.

 Output
yaml
Copy
Edit
Epoch 0: Loss = 0.6931, Accuracy = 0.62
Epoch 100: Loss = 0.45, Accuracy = 0.86
Epoch 200: Loss = 0.35, Accuracy = 0.90
...
Epoch 1000: Loss = 0.14, Accuracy = 0.97

Test Accuracy: 0.96

![image](https://github.com/user-attachments/assets/1e2c25ff-051f-4f7a-9acd-682d3a80b235)
LOSS CURVE
![image](https://github.com/user-attachments/assets/31f4a076-83df-48d2-b4cc-07b234931477)

