import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(a):
    return 1 - np.square(a)

# Binary cross-entropy loss
def compute_loss(y_true, y_pred):
    m = y_true.shape[1]
    return -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) / m

# Accuracy
def compute_accuracy(y_true, y_pred):
    y_pred_labels = y_pred > 0.5
    return np.mean(y_true == y_pred_labels)

# Load and preprocess the dataset
data = load_breast_cancer()
X = data.data
y = data.target.reshape(1, -1)  # Shape: (1, n_samples)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X).T  # Shape: (n_features, n_samples)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T
y_train, y_test = y_train.T, y_test.T

# Network architecture
n_x = X_train.shape[0]  # Input size (30)
n_h = 10                # Hidden layer size
n_y = 1                 # Output size

# Initialize weights and biases
np.random.seed(1)
W1 = np.random.randn(n_h, n_x) * 0.01
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h) * 0.01
b2 = np.zeros((n_y, 1))

# Training parameters
learning_rate = 0.01
num_epochs = 1000
losses = []

# Training loop
for epoch in range(num_epochs):
    # Forward propagation
    Z1 = np.dot(W1, X_train) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # Loss
    loss = compute_loss(y_train, A2)
    losses.append(loss)

    # Backward propagation
    m = X_train.shape[1]
    dZ2 = A2 - y_train
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * tanh_derivative(A1)
    dW1 = (1 / m) * np.dot(dZ1, X_train.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    # Update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Print loss every 100 epochs
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        acc = compute_accuracy(y_train, A2)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

# Evaluate on test set
Z1_test = np.dot(W1, X_test) + b1
A1_test = tanh(Z1_test)
Z2_test = np.dot(W2, A1_test) + b2
A2_test = sigmoid(Z2_test)

test_accuracy = compute_accuracy(y_test, A2_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Loss Curve During Training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
