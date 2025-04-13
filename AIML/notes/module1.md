# 📘 Introduction to Artificial Intelligence (AI)

## 1. Introduction to AI: Techniques, Applications, and AI Problems

### 🌟 What is AI?

Artificial Intelligence (AI) is the simulation of human intelligence by machines. It enables machines to perform tasks that typically require human intelligence.

### 🧠 Techniques in AI

- **Search algorithms** – Explore possible solutions.
- **Knowledge representation** – Represent information for reasoning.
- **Machine Learning (ML)** – Learn patterns from data.
- **Natural Language Processing (NLP)** – Understand/generate human language.
- **Computer Vision** – Interpret visual data.
- **Robotics** – Control of physical machines.

### 🔧 Applications of AI

- **Healthcare**: Medical diagnosis, drug discovery.
- **Finance**: Fraud detection, algorithmic trading.
- **Retail**: Recommendation systems.
- **Automobiles**: Self-driving cars.
- **Assistants**: Alexa, Siri, Google Assistant.

### ⚠️ AI Problems

- **Search problems** – e.g., pathfinding.
- **Game playing** – Chess, Go.
- **Planning** – Logistics, scheduling.
- **Diagnosis** – Fault finding.
- **Classification** – Spam detection.

---

## 2. Intelligent Agents: Structure, Types, Agent Environments

### 🧑‍💻 What is an Intelligent Agent?

An **agent** is an entity that perceives its environment through **sensors** and acts upon it using **actuators**.

### 🏗️ Structure of an Agent

- **Sensors** – Input from the environment.
- **Actuators** – Output to the environment.
- **Agent Function** – Maps percepts to actions.
- **Agent Program** – Implements the agent function.

### 🧩 Types of Agents

1. **Simple Reflex Agents** – React only to current input.
2. **Model-based Reflex Agents** – Use internal state.
3. **Goal-based Agents** – Act to achieve goals.
4. **Utility-based Agents** – Optimize a utility function.
5. **Learning Agents** – Improve performance over time.

### 🌍 Agent Environments

| Property      | Types                        |
| ------------- | ---------------------------- |
| Observability | Fully / Partially Observable |
| Determinism   | Deterministic / Stochastic   |
| Episodicity   | Episodic / Sequential        |
| Dynamics      | Static / Dynamic             |
| Discreteness  | Discrete / Continuous        |
| Agent Count   | Single-agent / Multi-agent   |

---

## 3. Understanding PEAS in AI

PEAS is a model for specifying the structure of intelligent agents.

### 🔍 PEAS = Performance, Environment, Actuators, Sensors

### 📌 Example: Self-driving Car

| PEAS Component  | Description                        |
| --------------- | ---------------------------------- |
| **Performance** | Safe driving, efficiency, legality |
| **Environment** | Roads, traffic, pedestrians        |
| **Actuators**   | Steering, throttle, brakes         |
| **Sensors**     | Cameras, GPS, radar, LIDAR         |

---

## 4. Problem Formulation and State Space Representation

### 🧩 Components of Problem Formulation

- **Initial State** – Starting point.
- **Goal State** – Desired outcome.
- **Actions** – Available operations.
- **Transition Model** – Result of an action.
- **Path Cost** – Cost from start to goal.

### 🔄 State Space Representation

A **state space** is a graph:

- **Nodes = States**
- **Edges = Actions leading from one state to another**

### 📌 Example: 8-Puzzle Problem

```
[1] [2] [3]
[4] [5] [6]
[7] [8] [ ]
```

- **Initial State**: Shuffled tiles.
- **Goal State**: Ordered tiles.
- **Actions**: Move blank tile (Up, Down, Left, Right).
- **Path Cost**: Number of moves.

---

## 🧠 Diagrams and Visual Aids

### 🖼️ Agent Structure

```
+------------+        +--------------+        +------------+
|  Sensors   | -----> | Agent Program| -----> | Actuators  |
+------------+        +--------------+        +------------+
       ↑                                         ↓
  Environment -----------------------------> Percepts & Actions
```

### 🧠 PEAS Framework (Simplified)

```
Performance ↔ Environment ↔ Sensors ↔ Agent ↔ Actuators
```

### 🔄 State Space (Graph Representation)

```
[Start] --Move1--> [State1] --Move2--> [Goal]
```

# 🤖 Introduction to Machine Learning (ML)

## 1. Introduction to ML

### 🧠 What is Machine Learning?

Machine Learning is a subset of Artificial Intelligence (AI) that gives systems the ability to learn and improve from experience without being explicitly programmed.

**Arthur Samuel's definition (1959):**

> "Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed."

### 📈 Types of Machine Learning

| Type                       | Description                                       | Examples                      |
| -------------------------- | ------------------------------------------------- | ----------------------------- |
| **Supervised Learning**    | Learn from labeled data (input → output)          | Spam detection, house pricing |
| **Unsupervised Learning**  | Find hidden patterns in unlabeled data            | Clustering, anomaly detection |
| **Reinforcement Learning** | Learn by trial and error using feedback (rewards) | Game AI, robotics             |

---

## 2. ML Techniques

### 🧪 Common ML Algorithms

- **Linear Regression** – Predict continuous values.
- **Logistic Regression** – Classification problems (e.g., yes/no).
- **Decision Trees & Random Forests** – Tree-based decision-making.
- **Support Vector Machines (SVM)** – Classification with hyperplanes.
- **K-Means Clustering** – Unsupervised grouping.
- **K-Nearest Neighbors (KNN)** – Classification by proximity.
- **Naive Bayes** – Based on Bayes' theorem.
- **Neural Networks** – Modeled after the human brain.

---

## 3. Introduction to Neural Networks (NN)

### 🧬 What is a Neural Network?

A Neural Network is a computational model inspired by the human brain. It consists of layers of nodes (neurons) that process inputs and produce outputs.

### 🧱 Structure of a Neural Network

```
Input Layer → Hidden Layers → Output Layer
```

Each neuron applies a **weighted sum + activation function**.

### 🔍 Common Activation Functions

- **Sigmoid** – Smooth S-curve output (0 to 1)
- **ReLU (Rectified Linear Unit)** – Output = max(0, x)
- **Tanh** – Output between -1 and 1

---

## 4. AI vs ML

| Feature      | Artificial Intelligence (AI)      | Machine Learning (ML)          |
| ------------ | --------------------------------- | ------------------------------ |
| **Goal**     | Simulate intelligence             | Learn from data                |
| **Scope**    | Broad (includes reasoning, logic) | Narrower (focuses on learning) |
| **Approach** | Rule-based or data-driven         | Primarily data-driven          |
| **Learning** | Not always present                | Core aspect                    |
| **Examples** | Expert systems, NLP               | Regression, clustering, SVMs   |

---

## 5. Applications of ML

### 🔧 Practical Uses of ML

| Domain                 | Applications                                  |
| ---------------------- | --------------------------------------------- |
| **Healthcare**         | Disease prediction, drug discovery            |
| **Finance**            | Fraud detection, algorithmic trading          |
| **Marketing**          | Customer segmentation, recommendation engines |
| **Retail**             | Inventory forecasting, personalization        |
| **Autonomous Systems** | Self-driving cars, drones                     |
| **NLP**                | Language translation, sentiment analysis      |

---

## 🧠 Visuals and Diagrams

### 🖼️ Machine Learning Types

```
                     +--------------------------+
                     |    Machine Learning      |
                     +--------------------------+
                       /          |            \
     Supervised   Unsupervised   Reinforcement
```

### 🧠 Neural Network Structure

```
[Input Layer] → [Hidden Layer 1] → [Hidden Layer 2] → [Output Layer]
```

### ⚖️ AI vs ML Relationship

```
+--------------------------+
|      Artificial          |
|      Intelligence        |
|  +--------------------+  |
|  |  Machine Learning  |  |
|  | +----------------+|  |
|  | |Deep Learning   ||  |
|  | +----------------+|  |
|  +--------------------+  |
+--------------------------+
```
