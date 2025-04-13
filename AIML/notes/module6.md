# Artificial Neural Networks: From Biological Foundations to Modern Architectures

## 1. Biological Inspiration: The Neuron

### 1.1 Structure of a Biological Neuron

Artificial Neural Networks (ANNs) draw inspiration from the structure and function of biological neurons in the human brain. A biological neuron consists of:

- **Dendrites**: Branch-like structures that receive signals from other neurons
- **Cell Body (Soma)**: Processes the incoming signals
- **Axon**: Transmits the processed signal to other neurons
- **Synapses**: Junction points where signals pass between neurons

When a neuron receives enough excitatory input to exceed its activation threshold, it "fires," sending an electrical impulse down its axon to communicate with other neurons.

### 1.2 The Artificial Neuron Analogy

An artificial neuron (or perceptron) mimics this biological process through:

- **Inputs (x₁, x₂, ..., xₙ)**: Analogous to signals received by dendrites
- **Weights (w₁, w₂, ..., wₙ)**: Represent the strength of synaptic connections
- **Summation function**: Aggregates weighted inputs (similar to the cell body)
- **Activation function**: Determines whether the neuron fires
- **Output**: The signal transmitted to other neurons

## 2. Neural Network Architecture and Types

### 2.1 Basic Architecture

Neural networks are organized in layers:

- **Input Layer**: Receives raw data
- **Hidden Layer(s)**: Processes information through weighted connections
- **Output Layer**: Produces the final result

Each layer consists of neurons (nodes) connected to neurons in adjacent layers.

### 2.2 Common Neural Network Types

#### 2.2.1 Feedforward Neural Networks (FNN)

Information flows in one direction from input to output without cycles or loops.

**Python Implementation:**

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train feedforward neural network
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', 
                   solver='adam', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Evaluate
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Visualize decision process for a 2D slice of the data
plt.figure(figsize=(12, 5))

# Plot decision boundaries using the first two features
plt.subplot(1, 2, 1)
h = 0.02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Create simplified data with only first two features for visualization
X_simplified = np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 8))]
Z = mlp.predict(X_simplified)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.title('Decision Boundary (First 2 Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot loss curve
plt.subplot(1, 2, 2)
plt.plot(mlp.loss_curve_)
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()
```

#### 2.2.2 Recurrent Neural Networks (RNN)

Contains loops to allow information persistence, useful for sequential data.

#### 2.2.3 Convolutional Neural Networks (CNN)

Specialized for grid-like data such as images, using convolutional layers.

#### 2.2.4 Self-Organizing Maps (SOM)

Unsupervised learning networks that produce low-dimensional representations of input data.

## 3. MP Neuron Model (McCulloch-Pitts)

The McCulloch-Pitts neuron is one of the earliest neural models, developed in 1943.

### 3.1 Characteristics

- **Binary inputs and outputs** (0 or 1)
- **Fixed threshold**
- **Equal weights** for all inputs
- **No learning capability**

### 3.2 Mathematical Model

The MP neuron computes the sum of its inputs and fires if the sum exceeds a threshold:

y = {
    1 if ∑(xᵢ) ≥ threshold
    0 otherwise
}

### 3.3 Python Implementation

```python
import numpy as np

class MPNeuron:
    def __init__(self, threshold):
        self.threshold = threshold
        
    def activate(self, inputs):
        """Compute the output of the MP neuron"""
        return 1 if sum(inputs) >= self.threshold else 0
    
    def predict(self, X):
        """Predict for multiple input samples"""
        return [self.activate(x) for x in X]

# Example: MP neuron implementation of the AND gate
mp_neuron = MPNeuron(threshold=2)

# Test with all possible input combinations
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
predictions = mp_neuron.predict(X)

# Display results
for inputs, prediction in zip(X, predictions):
    print(f"Inputs: {inputs}, Output: {prediction}")
```

### 3.4 Limitations

- Cannot implement XOR function (not linearly separable)
- No learning mechanism
- Binary only
- All inputs weighted equally

## 4. Hebb Network

Developed by Donald Hebb in 1949, the Hebb network is based on the principle that neurons that fire together, wire together.

### 4.1 Hebbian Learning Rule

The Hebbian learning rule adjusts weights based on the correlation between input and output:

Δwᵢⱼ = η × xᵢ × yⱼ

Where:
- Δwᵢⱼ is the change in weight
- η is the learning rate
- xᵢ is the input
- yⱼ is the output

### 4.2 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class HebbNet:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.learning_rate = learning_rate
        
    def predict(self, x):
        """Calculate the output"""
        activation = np.dot(x, self.weights)
        return 1 if activation >= 0 else -1
    
    def train(self, X, y, epochs=1):
        """Train the Hebbian network"""
        weight_history = [self.weights.copy()]
        
        for _ in range(epochs):
            for i in range(len(X)):
                x = X[i]
                output = self.predict(x)
                
                # Apply Hebbian learning rule
                delta_w = self.learning_rate * np.outer(x, output).flatten()
                self.weights += delta_w
                weight_history.append(self.weights.copy())
                
        return weight_history

# Example: Pattern association with Hebbian learning
# Define two patterns
pattern1 = np.array([1, 1, -1, -1])
pattern2 = np.array([-1, -1, 1, 1])

X = np.vstack([pattern1, pattern2])
y = np.array([1, -1])  # Associated outputs

# Train the Hebbian network
hebb_net = HebbNet(input_size=4, learning_rate=0.1)
weight_history = hebb_net.train(X, y, epochs=1)

# Test the network
for i, pattern in enumerate(X):
    output = hebb_net.predict(pattern)
    print(f"Pattern {i+1}: {pattern} -> Output: {output}")

# Visualize weight changes
plt.figure(figsize=(10, 6))
weight_history = np.array(weight_history)
for i in range(weight_history.shape[1]):
    plt.plot(weight_history[:, i], label=f'Weight {i+1}')
    
plt.title('Weight Changes During Hebbian Learning')
plt.xlabel('Update Step')
plt.ylabel('Weight Value')
plt.legend()
plt.grid(True)
plt.show()
```

### 4.3 Limitations

- Can cause unlimited weight growth without normalization
- Limited in capacity and capability
- No error correction mechanism

## 5. Learning Rules

### 5.1 Perceptron Learning Rule

The perceptron learning rule adjusts weights based on the error between the desired output and the actual output.

#### 5.1.1 Algorithm

1. Initialize weights to small random values
2. For each training example:
   - Calculate the output using current weights
   - Update weights if output doesn't match target: wᵢ(new) = wᵢ(old) + η(d - y)xᵢ
     - Where d is the desired output and y is the actual output
3. Repeat until all examples are correctly classified or max iterations reached

#### 5.1.2 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def activate(self, x):
        """Step function"""
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        """Generate predictions"""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activate(x) for x in linear_output])
    
    def fit(self, X, y):
        """Train the perceptron"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Error history
        self.errors = []
        
        # Learning
        for _ in range(self.n_iterations):
            errors = 0
            
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                
                # Predict
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activate(linear_output)
                
                # Update weights if prediction is wrong
                if y_i != y_pred:
                    update = self.lr * (y_i - y_pred)
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            
            self.errors.append(errors)
            
            # Stop if converged
            if errors == 0:
                break

# Generate linearly separable data
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
y = np.where(y == 0, 0, 1)  # Convert to binary 0/1

# Train perceptron
perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
perceptron.fit(X, y)

# Visualize the results
plt.figure(figsize=(12, 5))

# Plot data and decision boundary
plt.subplot(1, 2, 1)
for label in np.unique(y):
    plt.scatter(X[y == label, 0], X[y == label, 1], label=f'Class {label}')

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.title('Perceptron Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot error history
plt.subplot(1, 2, 2)
plt.plot(perceptron.errors)
plt.title('Perceptron Error History')
plt.xlabel('Iterations')
plt.ylabel('Number of Misclassifications')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 5.2 Delta Rule (Widrow-Hoff)

Also known as the Least Mean Square (LMS) rule, the delta rule minimizes the squared error between the desired output and the actual output.

#### 5.2.1 Algorithm

For continuous output neurons, the weight update is:

Δwᵢ = η(d - y)xᵢ

Where:
- η is the learning rate
- d is the desired output
- y is the actual output
- xᵢ is the input

#### 5.2.2 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class DeltaRuleNeuron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def activate(self, x):
        """Linear activation function"""
        return x
    
    def predict(self, X):
        """Generate predictions"""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activate(x) for x in linear_output])
    
    def fit(self, X, y):
        """Train using the delta rule"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Error history
        self.errors = []
        
        # Learning
        for _ in range(self.n_iterations):
            total_error = 0
            
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                
                # Predict
                y_pred = self.activate(np.dot(x_i, self.weights) + self.bias)
                
                # Calculate error
                error = y_i - y_pred
                total_error += error**2
                
                # Update weights using delta rule
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error
            
            # Record mean squared error
            mse = total_error / n_samples
            self.errors.append(mse)
            
            # Convergence check
            if mse < 0.0001:
                break

# Generate regression data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 0.5  # y = 2x + 1 + noise

# Train delta rule neuron
delta_neuron = DeltaRuleNeuron(learning_rate=0.001, n_iterations=100)
delta_neuron.fit(X, y)

# Visualize the results
plt.figure(figsize=(12, 5))

# Plot data and fitted line
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.7, label='Data points')
x_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = delta_neuron.predict(x_line)
plt.plot(x_line, y_line, 'r-', label='Fitted line')
plt.title('Delta Rule Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Plot error history
plt.subplot(1, 2, 2)
plt.plot(delta_neuron.errors)
plt.title('Delta Rule Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 5.3 Backpropagation Algorithm

Backpropagation is the core learning algorithm for multilayer neural networks, propagating error gradients backward through the network to update weights.

#### 5.3.1 Algorithm

1. **Forward Pass**: Compute outputs of all neurons layer by layer
2. **Compute Error**: Calculate the difference between actual and desired outputs
3. **Backward Pass**: Propagate error gradients backward through the network
4. **Update Weights**: Adjust weights in proportion to their contribution to the error

#### 5.3.2 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
            
        return self.activations[-1]
    
    def backward(self, X, y):
        """Backward pass to update weights"""
        m = X.shape[0]
        output = self.activations[-1]
        
        # Output layer error
        delta = (output - y) * self.sigmoid_derivative(output)
        
        # Backpropagate the error
        for i in range(len(self.weights) - 1, -1, -1):
            # Update weights and biases
            self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, delta) / m
            self.biases[i] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True) / m
            
            # Compute error for previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
    
    def train(self, X, y, epochs=1000):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = np.mean(np.square(output - y))
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.6f}')
        
        return losses
    
    def predict(self, X):
        """Generate predictions"""
        return self.forward(X)

# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train network
nn = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=0.5)
losses = nn.train(X, y, epochs=5000)

# Test predictions
predictions = nn.predict(X)
print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {predictions[i][0]:.4f}")

# Visualize learning
plt.figure(figsize=(12, 5))

# Plot loss history
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Neural Network Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.grid(True)

# Visualize decision boundary
plt.subplot(1, 2, 2)
h = 0.01
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), edgecolors='k', cmap=plt.cm.RdBu)
plt.title('Decision Boundary for XOR Problem')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

## 6. Self-Organizing Maps (SOM)

Self-Organizing Maps (SOMs), also known as Kohonen maps, are a type of artificial neural network that use unsupervised learning to produce a low-dimensional representation of the input space, making them useful for visualization and clustering.

### 6.1 Architecture and Algorithm

- SOM consists of a grid of neurons, each with a weight vector of the same dimension as the input data
- During training, neurons compete to respond to input patterns
- The winning neuron and its neighbors adjust their weights to better match the input

### 6.2 Training Process

1. Initialize weight vectors randomly
2. For each input vector:
   - Find the Best Matching Unit (BMU) - the neuron with weights closest to the input
   - Update the BMU and its neighbors to make them more similar to the input
3. Decrease the learning rate and neighborhood radius over time

### 6.3 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class SOM:
    def __init__(self, x_dim, y_dim, input_dim, learning_rate=0.1, sigma=None):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(x_dim, y_dim) / 2
        
        # Initialize weights
        self.weights = np.random.rand(x_dim, y_dim, input_dim)
        
        # Initialize grid coordinates
        self.grid = np.array([[[i, j] for j in range(y_dim)] for i in range(x_dim)])
    
    def find_bmu(self, x):
        """Find the Best Matching Unit (BMU) for input x"""
        distances = np.sum((self.weights - x) ** 2, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)
    
    def update_weights(self, x, iteration, max_iterations):
        """Update weights based on input and current iteration"""
        # Find BMU
        bmu = self.find_bmu(x)
        
        # Calculate decay parameters
        t = iteration / max_iterations
        lr = self.learning_rate * np.exp(-t)
        sigma = self.sigma * np.exp(-t)
        
        # Calculate neighborhood function
        distances = np.sum((self.grid - np.array([bmu])) ** 2, axis=2)
        neighborhood = np.exp(-distances / (2 * sigma**2))
        
        # Update weights
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                self.weights[i, j] += lr * neighborhood[i, j] * (x - self.weights[i, j])
    
    def train(self, X, iterations=100):
        """Train the SOM"""
        for iter in range(iterations):
            if iter % 10 == 0:
                print(f"Iteration {iter}/{iterations}")
            
            # Pick a random sample
            x = X[np.random.randint(0, X.shape[0])]
            
            # Update weights
            self.update_weights(x, iter, iterations)
    
    def get_cluster_assignments(self, X):
        """Get cluster assignments for data points"""
        clusters = []
        for x in X:
            bmu = self.find_bmu(x)
            clusters.append(bmu)
        return np.array(clusters)

# Generate sample data
X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.8, random_state=42, n_features=2)

# Scale data to [0, 1]
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Train SOM
som = SOM(10, 10, 2, learning_rate=0.5, sigma=2)
som.train(X, iterations=1000)

# Get cluster assignments
clusters = som.get_cluster_assignments(X)

# Visualize data and SOM grid
plt.figure(figsize=(12, 10))

# Plot original data
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot SOM grid
plt.subplot(2, 2, 2)
for i in range(som.x_dim):
    for j in range(som.y_dim):
        plt.plot(som.weights[i, j, 0], som.weights[i, j, 1], 'k.', markersize=8)
        if i < som.x_dim - 1:
            plt.plot([som.weights[i, j, 0], som.weights[i+1, j, 0]],
                     [som.weights[i, j, 1], som.weights[i+1, j, 1]], 'k-', alpha=0.3)
        if j < som.y_dim - 1:
            plt.plot([som.weights[i, j, 0], som.weights[i, j+1, 0]],
                     [som.weights[i, j, 1], som.weights[i, j+1, 1]], 'k-', alpha=0.3)

plt.title('SOM Grid')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot data with cluster assignments
plt.subplot(2, 2, 3)
unique_clusters = np.unique(clusters, axis=0)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
cluster_map = {tuple(c): colors[i] for i, c in enumerate(unique_clusters)}

for point, cluster in zip(X, clusters):
    plt.scatter(point[0], point[1], color=cluster_map[tuple(cluster)], alpha=0.7)

plt.title('Data with Cluster Assignments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot U-Matrix (Unified Distance Matrix) - shows cluster boundaries
plt.subplot(2, 2, 4)
u_matrix = np.zeros((som.x_dim, som.y_dim))
for i in range(som.x_dim):
    for j in range(som.y_dim):
        # Calculate average distance to neighbors
        neighbors = []
        if i > 0:
            neighbors.append(som.weights[i-1, j])
        if i < som.x_dim - 1:
            neighbors.append(som.weights[i+1, j])
        if j > 0:
            neighbors.append(som.weights[i, j-1])
        if j < som.y_dim - 1:
            neighbors.append(som.weights[i, j+1])
        
        u_matrix[i, j] = np.mean([np.linalg.norm(som.weights[i, j] - neighbor) for neighbor in neighbors])

plt.imshow(u_matrix, cmap='viridis', interpolation='none')
plt.colorbar(label='Average Distance to Neighbors')
plt.title('U-Matrix (Cluster Boundaries)')
plt.tight_layout()
plt.show()
```

### 6.4 Applications of SOMs

1. **Data Visualization**: Reduce high-dimensional data to 2D or 3D for visualization
2. **Clustering**: Group similar data points without supervision
3. **Feature Detection**: Identify important features in the input space
4. **Pattern Recognition**: Recognize patterns in complex data

### 6.5 Advantages and Disadvantages

**Advantages:**
- Preserves topological relationships in the input data
- Reduces dimensions while preserving important features
- Works well with unlabeled data
- Can detect clusters of arbitrary shapes

**Disadvantages:**
- Requires specifying grid dimensions in advance
- Results depend on initialization and training parameters
- Training can be computationally expensive
- Difficulty interpreting the meaning of grid positions

## 7. Comparative Analysis of Neural Models

| Model | Year | Input Type | Learning Type | Applications | Key Features |
|-------|------|------------|--------------|--------------|--------------|
| MP Neuron | 1943 | Binary | None | Logic gates | First computational neuron model |
| Hebb Network | 1949 | Real | Unsupervised | Pattern association | "Neurons that fire together, wire together" |
| Perceptron | 1958 | Real | Supervised | Binary classification | Convergence guarantee for linearly separable data |
| Delta Rule | 1960 | Real | Supervised | Regression, Classification | Minimizes squared error |
| Backpropagation | 1986 | Real | Supervised | Various ML tasks | Allows training deep networks |
| SOM | 1982 | Real | Unsupervised | Clustering, Visualization |

## 7. Comparative Analysis of Neural Models (Continued)

| Model | Year | Input Type | Learning Type | Applications | Key Features |
|-------|------|------------|--------------|--------------|--------------|
| MP Neuron | 1943 | Binary | None | Logic gates | First computational neuron model |
| Hebb Network | 1949 | Real | Unsupervised | Pattern association | "Neurons that fire together, wire together" |
| Perceptron | 1958 | Real | Supervised | Binary classification | Convergence guarantee for linearly separable data |
| Delta Rule | 1960 | Real | Supervised | Regression, Classification | Minimizes squared error |
| Backpropagation | 1986 | Real | Supervised | Various ML tasks | Allows training deep networks |
| SOM | 1982 | Real | Unsupervised | Clustering, Visualization | Topology-preserving mapping |

## 8. Advanced Neural Network Concepts

### 8.1 Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

#### 8.1.1 Common Activation Functions

1. **Sigmoid**: σ(x) = 1/(1 + e^(-x))
   - Outputs between 0 and 1
   - Historically popular but suffers from vanishing gradient problem

2. **Hyperbolic Tangent (tanh)**: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
   - Outputs between -1 and 1
   - Zero-centered but still has vanishing gradient issues

3. **Rectified Linear Unit (ReLU)**: f(x) = max(0, x)
   - Simple and computationally efficient
   - Helps mitigate vanishing gradient problem
   - Can suffer from "dying ReLU" problem (neurons permanently deactivated)

4. **Leaky ReLU**: f(x) = max(αx, x) where α is a small constant
   - Addresses the dying ReLU problem

5. **Softmax**: Used for multi-class classification in the output layer
   - Converts outputs to probabilities that sum to 1

#### 8.1.2 Comparison of Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

# Generate x values
x = np.linspace(-5, 5, 1000)

# Plot activation functions
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x, tanh(x))
plt.title('Hyperbolic Tangent (tanh)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU (α=0.01)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 8.2 Regularization Techniques

Regularization methods help prevent overfitting by constraining the model's complexity.

#### 8.2.1 Common Regularization Methods

1. **L1 Regularization (Lasso)**
   - Adds |w| term to the loss function
   - Encourages sparse weight matrices by driving some weights to zero

2. **L2 Regularization (Ridge)**
   - Adds ||w||² term to the loss function
   - Discourages large weights

3. **Dropout**
   - Randomly deactivates neurons during training
   - Forces the network to learn redundant representations

4. **Early Stopping**
   - Stops training when validation error starts increasing
   - Prevents the model from overfitting to the training data

#### 8.2.2 Dropout Implementation

```python
import numpy as np

class NeuralNetworkWithDropout:
    def __init__(self, layer_sizes, learning_rate=0.1, dropout_rate=0.2):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X, is_training=True):
        """Forward pass through the network with dropout"""
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.sigmoid(z)
            
            # Apply dropout during training
            if is_training and i < len(self.weights) - 1:  # No dropout on output layer
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape) / (1 - self.dropout_rate)
                self.dropout_masks.append(dropout_mask)
                a *= dropout_mask
            else:
                self.dropout_masks.append(None)
                
            self.activations.append(a)
            
        return self.activations[-1]
    
    def backward(self, X, y):
        """Backward pass to update weights"""
        m = X.shape[0]
        output = self.activations[-1]
        
        # Output layer error
        delta = (output - y) * self.sigmoid_derivative(output)
        
        # Backpropagate the error
        for i in range(len(self.weights) - 1, -1, -1):
            # Update weights and biases
            self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, delta) / m
            self.biases[i] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True) / m
            
            # Compute error for previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
                # Apply dropout mask
                if self.dropout_masks[i-1] is not None:
                    delta *= self.dropout_masks[i-1]
    
    def train(self, X, y, epochs=1000, batch_size=32):
        """Train the neural network with mini-batch gradient descent"""
        n_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass (with dropout)
                output = self.forward(X_batch, is_training=True)
                
                # Backward pass
                self.backward(X_batch, y_batch)
            
            # Compute loss on full dataset (without dropout)
            output = self.forward(X, is_training=False)
            loss = np.mean(np.square(output - y))
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.6f}')
        
        return losses
    
    def predict(self, X):
        """Generate predictions (without dropout)"""
        return self.forward(X, is_training=False)
```

### 8.3 Weight Initialization Strategies

Proper weight initialization is crucial for training deep neural networks effectively.

#### 8.3.1 Common Initialization Methods

1. **Zeros or Constant Initialization**
   - Not recommended as it makes neurons learn the same features

2. **Random Initialization**
   - Basic approach that breaks symmetry
   - May lead to vanishing/exploding gradients

3. **Xavier/Glorot Initialization**
   - Designed for sigmoid/tanh activations
   - Weights ~ N(0, 2/(n_in + n_out))

4. **He Initialization**
   - Designed for ReLU activations
   - Weights ~ N(0, 2/n_in)

#### 8.3.2 Implementation of Weight Initialization Methods

```python
import numpy as np

def initialize_weights(method, layer_sizes):
    """Initialize weights using different strategies"""
    weights = []
    
    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i+1]
        
        if method == "zeros":
            w = np.zeros((n_in, n_out))
        
        elif method == "random":
            w = np.random.randn(n_in, n_out) * 0.01
        
        elif method == "xavier":
            limit = np.sqrt(6 / (n_in + n_out))
            w = np.random.uniform(-limit, limit, (n_in, n_out))
        
        elif method == "he":
            std = np.sqrt(2 / n_in)
            w = np.random.randn(n_in, n_out) * std
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        weights.append(w)
    
    return weights

# Example usage
layer_sizes = [784, 128, 64, 10]  # Example sizes for MNIST classification
weights_xavier = initialize_weights("xavier", layer_sizes)
weights_he = initialize_weights("he", layer_sizes)

# Visualize weight distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(weights_xavier[0].flatten(), bins=50, alpha=0.7)
plt.title('Xavier/Glorot Initialization')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(weights_he[0].flatten(), bins=50, alpha=0.7)
plt.title('He Initialization')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## 9. Implementation of a Complete Neural Network from Scratch

Let's implement a complete multi-layer neural network for a real problem from scratch using the concepts we've covered.

### 9.1 Neural Network Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

class NeuralNetwork:
    def __init__(self, layer_sizes, activations, weight_init="he"):
        """
        Initialize neural network
        
        Parameters:
        - layer_sizes: List of neurons in each layer (input, hidden, output)
        - activations: List of activation functions for each layer
        - weight_init: Weight initialization strategy
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activations = activations
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            
            # Weight initialization
            if weight_init == "xavier":
                limit = np.sqrt(6 / (n_in + n_out))
                w = np.random.uniform(-limit, limit, (n_in, n_out))
            elif weight_init == "he":
                std = np.sqrt(2 / n_in)
                w = np.random.randn(n_in, n_out) * std
            else:
                w = np.random.randn(n_in, n_out) * 0.01
                
            b = np.zeros((1, n_out))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def activate(self, x, activation):
        """Apply activation function"""
        if activation == "sigmoid":
            return self.sigmoid(x)
        elif activation == "relu":
            return self.relu(x)
        elif activation == "softmax":
            return self.softmax(x)
        elif activation == "linear":
            return x
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def activate_derivative(self, x, activation):
        """Apply derivative of activation function"""
        if activation == "sigmoid":
            return self.sigmoid_derivative(x)
        elif activation == "relu":
            return self.relu_derivative(x)
        elif activation == "softmax" or activation == "linear":
            return 1  # Handled differently in backpropagation
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def one_hot_encode(self, y, num_classes):
        """Convert class labels to one-hot encoding"""
        return np.eye(num_classes)[y]
    
    def forward(self, X):
        """Forward pass through the network"""
        self.z_values = []
        self.a_values = [X]  # Input layer activation
        
        # Propagate through hidden layers
        for i in range(self.num_layers - 1):
            z = np.dot(self.a_values[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            a = self.activate(z, self.activations[i])
            self.a_values.append(a)
            
        return self.a_values[-1]  # Output layer activation
    
    def backward(self, X, y, learning_rate=0.01, reg_lambda=0.01):
        """Backward pass to update weights using gradient descent with L2 regularization"""
        m = X.shape[0]  # Number of samples
        
        # Convert to one-hot if needed
        if len(y.shape) == 1:
            y = self.one_hot_encode(y, self.layer_sizes[-1])
        
        # Backpropagation
        deltas = [None] * (self.num_layers - 1)
        
        # Output layer error (special case for softmax + cross-entropy)
        if self.activations[-1] == "softmax":
            deltas[-1] = self.a_values[-1] - y
        else:
            deltas[-1] = (self.a_values[-1] - y) * self.activate_derivative(self.a_values[-1], self.activations[-1])
        
        # Hidden layers error
        for l in range(self.num_layers - 2, 0, -1):
            deltas[l-1] = np.dot(deltas[l], self.weights[l].T) * self.activate_derivative(self.a_values[l], self.activations[l-1])
        
        # Update weights and biases
        for l in range(self.num_layers - 2, -1, -1):
            # L2 regularization term
            reg_term = reg_lambda * self.weights[l]
            
            # Weight update
            self.weights[l] -= learning_rate * (np.dot(self.a_values[l].T, deltas[l]) / m + reg_term)
            
            # Bias update (no regularization for bias)
            self.biases[l] -= learning_rate * np.sum(deltas[l], axis=0, keepdims=True) / m
    
    def compute_loss(self, y_true, y_pred, reg_lambda=0.01):
        """Compute loss with L2 regularization"""
        m = y_true.shape[0]
        
        # Convert to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = self.one_hot_encode(y_true, self.layer_sizes[-1])
        
        # Cross-entropy loss
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cross_entropy = -np.sum(y_true * np.log(y_pred)) / m
        
        # L2 regularization term
        l2_reg = 0
        for w in self.weights:
            l2_reg += np.sum(np.square(w))
        l2_reg *= reg_lambda / (2 * m)
        
        return cross_entropy + l2_reg
    
    def train(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32, 
              learning_rate=0.01, reg_lambda=0.01, early_stopping=True, patience=10):
        """Train the neural network using mini-batch gradient descent"""
        m = X.shape[0]
        train_losses = []
        val_losses = []
        
        # For early stopping
        best_val_loss = float('inf')
        wait = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass
                self.backward(X_batch, y_batch, learning_rate, reg_lambda)
            
            # Compute training loss
            y_pred_train = self.forward(X)
            train_loss = self.compute_loss(y, y_pred_train, reg_lambda)
            train_losses.append(train_loss)
            
            # Compute validation loss if provided
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward(X_val)
                val_loss = self.compute_loss(y_val, y_pred_val, reg_lambda)
                val_losses.append(val_loss)
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                    else:
                        wait += 1
                    
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            # Print progress
            if epoch % 10 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}")
        
        return {"train_loss": train_losses, "val_loss": val_losses}
    
    def predict(self, X):
        """Generate class predictions"""
        y_prob = self.forward(X)
        return np.argmax(y_prob, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate the model performance"""
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        
        return {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix
        }

# Example usage with digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define model architecture
input_size = X.shape[1]  # 64 for digits
hidden_size1 = 128
hidden_size2 = 64
output_size = 10  # 10 digit classes

# Create neural network
nn = NeuralNetwork(
    layer_sizes=[input_size, hidden_size1, hidden_size2, output_size],
    activations=["relu", "relu", "softmax"],
    weight_init="he"
)

# Train the model
history = nn.train(
    X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=200,
    batch_size=32,
    learning_rate=0.01,
    reg_lambda=0.001
)

# Evaluate on test set
results = nn.evaluate(X_test, y_test)
print(f"Test accuracy: {results['accuracy']:.4f}")

# Plot learning curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot confusion matrix
plt.subplot(1, 2, 2)
plt.imshow(results['confusion_matrix'], cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

# Visualize some predictions
plt.figure(figsize=(12, 6))
predictions = nn.predict(X_test)

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"True: {y_test[i]}, Pred: {predictions[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

## 10. Applications and Future Directions

### 10.1 Key Application Areas

1. **Computer Vision**
   - Image classification, object detection, segmentation
   - Face recognition, medical image analysis

2. **Natural Language Processing**
   - Text classification, sentiment analysis
   - Machine translation, chatbots

3. **Time Series Analysis**
   - Stock price prediction, weather forecasting
   - Anomaly detection

4. **Reinforcement Learning**
   - Game playing (Chess, Go, video games)
   - Robotics, autonomous vehicles

### 10.2 Future Directions

1. **Energy-Efficient Neural Networks**
   - Reducing the computational and energy requirements

2. **Neuromorphic Computing**
   - Hardware designs inspired by the brain's architecture

3. **Explainable AI**
   - Making neural network decisions more interpretable

4. **Neuro-Symbolic AI**
   - Combining neural networks with symbolic reasoning

5. **Continual Learning**
   - Allowing networks to learn continuously without forgetting

## 11. Conclusion

Artificial Neural Networks have evolved significantly from their early biological inspiration. From simple models like the MP Neuron and Perceptron to complex architectures with advanced learning algorithms, neural networks have become powerful tools for solving complex problems across various domains.

Understanding the fundamentals of neural computation, different network architectures, and learning algorithms provides a solid foundation for applying these techniques to real-world problems. As research continues to advance, neural networks will likely become even more capable, efficient, and integral to artificial intelligence systems.
