# Machine Learning Fundamentals

## Learning Paradigms

### Supervised Learning

Supervised learning is a paradigm where an algorithm learns from labeled training data to make predictions or decisions.

**Key Characteristics:**
- Requires labeled input-output pairs
- Goal is to learn a mapping function from inputs to outputs
- Used for classification and regression tasks

**Examples:**
- Email spam detection (classification)
- House price prediction (regression)
- Image recognition (classification)

### Unsupervised Learning

Unsupervised learning works with unlabeled data to identify patterns, structures, or relationships.

**Key Characteristics:**
- Works with unlabeled data
- Goal is to discover hidden patterns or data groupings
- No "correct answers" provided during training

**Examples:**
- Customer segmentation
- Anomaly detection
- Dimensionality reduction

### Reinforcement Learning

Reinforcement learning involves an agent learning to make decisions by performing actions in an environment to maximize cumulative rewards.

**Key Characteristics:**
- Based on reward/penalty feedback
- Agent learns through trial and error
- Balances exploration and exploitation

**Components:**
- Agent: The learning algorithm
- Environment: What the agent interacts with
- Actions: What the agent can do
- States: Situations the agent can be in
- Rewards: Feedback from the environment

**Applications:**
- Game playing (AlphaGo)
- Robotics
- Autonomous vehicles
- Resource management

## Steps in Designing ML Systems

1. **Problem Definition**
   - Define the problem clearly
   - Identify business objectives
   - Determine if ML is the right approach

2. **Data Collection**
   - Gather relevant data from various sources
   - Ensure data quality and quantity
   - Consider data privacy and legal aspects

3. **Data Preprocessing**
   - Clean data (handle missing values, outliers)
   - Transform data (normalization, encoding)
   - Feature engineering
   - Split data into training, validation, and test sets

4. **Model Selection**
   - Choose appropriate algorithms based on problem type
   - Consider model complexity vs. available data
   - Balance interpretability and performance

5. **Training**
   - Fit models on training data
   - Tune hyperparameters
   - Use cross-validation

6. **Evaluation**
   - Assess model performance on validation set
   - Use appropriate metrics (accuracy, F1-score, etc.)
   - Address overfitting/underfitting

7. **Deployment**
   - Integrate model into production systems
   - Set up monitoring and logging
   - Plan for model updates

8. **Monitoring and Maintenance**
   - Track model performance over time
   - Detect concept drift
   - Retrain as needed

## Machine Learning Algorithms

### Linear Regression

Linear regression models the relationship between a dependent variable and one or more independent variables using a linear equation.

**Mathematical Representation:**
- For simple linear regression: y = β₀ + β₁x + ε
- For multiple linear regression: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Where:
- y is the dependent variable
- x₁, x₂, ..., xₙ are independent variables
- β₀, β₁, ..., βₙ are the coefficients
- ε is the error term

**Learning Objective:**
Minimize the sum of squared errors (SSE):
- SSE = Σ(yᵢ - ŷᵢ)²

**Applications:**
- Sales forecasting
- Risk assessment
- Trend analysis

### Support Vector Machine (SVM)

SVM is a supervised learning algorithm that finds the optimal hyperplane to separate different classes in a high-dimensional space.

**Key Concepts:**
- **Hyperplane:** Decision boundary that separates classes
- **Margin:** Distance between the hyperplane and the nearest data points
- **Support Vectors:** Data points closest to the hyperplane
- **Kernel Trick:** Method to transform data into higher dimensions

**Types:**
- Linear SVM
- Non-linear SVM (using kernel functions)
- Soft-margin SVM (allowing some misclassifications)

**Common Kernels:**
- Linear: K(x, y) = x^T y
- Polynomial: K(x, y) = (γx^T y + r)^d
- RBF (Gaussian): K(x, y) = exp(-γ||x - y||²)

**Applications:**
- Text classification
- Image recognition
- Bioinformatics

### Bayesian Belief Network

A Bayesian Belief Network (BBN) is a probabilistic graphical model that represents variables and their conditional dependencies via a directed acyclic graph (DAG).

**Components:**
- **Nodes:** Represent random variables
- **Edges:** Represent conditional dependencies
- **Conditional Probability Tables (CPTs):** Quantify relationships

**Properties:**
- Models cause-effect relationships
- Handles uncertainty and incomplete data
- Combines prior knowledge with observed data

**Inference Methods:**
- Exact inference (junction tree, variable elimination)
- Approximate inference (sampling methods, variational inference)

**Applications:**
- Medical diagnosis
- Risk analysis
- Decision support systems

### Decision Tree with Gini Index

Decision trees are hierarchical models that make decisions based on a series of questions.

**Components:**
- **Root Node:** Starting point
- **Internal Nodes:** Decision points
- **Leaf Nodes:** Final outcomes/predictions
- **Branches:** Possible outcomes of a decision

**Gini Index:**
- Measures the impurity or probability of incorrect classification
- Gini(D) = 1 - Σⱼ(pⱼ)²
  - Where pⱼ is the probability of class j

**Building Process:**
1. Start with root node containing all data
2. Calculate Gini Index for each potential split
3. Choose attribute with lowest Gini Index value
4. Split data based on selected attribute
5. Recursively repeat for each child node
6. Stop when a termination condition is met

**Advantages:**
- Easy to understand and interpret
- Requires minimal data preprocessing
- Can handle both numerical and categorical data

**Disadvantages:**
- Prone to overfitting
- Biased toward attributes with more levels
- Can be unstable with small variations in data

## Performance Metrics

### Confusion Matrix

A confusion matrix is a table used to describe the performance of a classification model by comparing predicted values against actual values.

**Structure:**

|                   | Predicted Positive | Predicted Negative |
|-------------------|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

- **True Positive (TP):** Correctly predicted positive class
- **True Negative (TN):** Correctly predicted negative class
- **False Positive (FP):** Incorrectly predicted positive class (Type I error)
- **False Negative (FN):** Incorrectly predicted negative class (Type II error)

### Accuracy

Accuracy measures the proportion of correct predictions among the total number of cases examined.

**Formula:**
- Accuracy = (TP + TN) / (TP + TN + FP + FN)

**When to Use:**
- Balanced datasets
- When all classes are equally important

**Limitations:**
- Misleading for imbalanced datasets

### Precision

Precision measures the proportion of correct positive identifications.

**Formula:**
- Precision = TP / (TP + FP)

**When to Use:**
- When the cost of false positives is high
- Ex: Spam detection, medical diagnosis

### Recall (Sensitivity)

Recall measures the proportion of actual positives that were correctly identified.

**Formula:**
- Recall = TP / (TP + FN)

**When to Use:**
- When the cost of false negatives is high
- Ex: Disease detection, fraud detection

### F-Score

F-Score is the harmonic mean of precision and recall, providing a balance between the two metrics.

**Formula:**
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Variations:**
- F₁-Score: Equal weight to precision and recall
- F₂-Score: More weight to recall
- F₀.₅-Score: More weight to precision

**When to Use:**
- When you need a balance between precision and recall
- For imbalanced datasets where accuracy might be misleading

### Sensitivity and Specificity

- **Sensitivity (Recall):** Ability to correctly identify positive cases
  - Sensitivity = TP / (TP + FN)

- **Specificity:** Ability to correctly identify negative cases
  - Specificity = TN / (TN + FP)

**ROC Curve:**
- Plots True Positive Rate (Sensitivity) vs. False Positive Rate (1-Specificity)
- AUC (Area Under Curve): Aggregate measure of performance across all classification thresholds
  - AUC = 1: Perfect classifier
  - AUC = 0.5: No better than random guessing
