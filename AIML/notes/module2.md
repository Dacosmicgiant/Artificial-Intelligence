# Problem Solving Agents

## 🔍 Uninformed Search Algorithms

Uninformed search algorithms explore the search space without any domain-specific knowledge. They are also known as **blind search** techniques.

We’ll explore the following:

- **Depth First Search (DFS)**
- **Breadth First Search (BFS)**
- **Uniform Cost Search (UCS)**
- **Depth Limited Search (DLS)**
- **Iterative Deepening Depth First Search (IDDFS)**

---

## 1. Depth First Search (DFS)

### 🔹 Concept:

Explores as far down a branch as possible before backtracking.

### 🧠 Strategy:

LIFO (Last In, First Out) – implemented using a stack.

### 💻 Python Code Example:

```python
def dfs(graph, start, goal, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    if start == goal:
        return [start]
    for neighbor in graph[start]:
        if neighbor not in visited:
            path = dfs(graph, neighbor, goal, visited)
            if path:
                return [start] + path
    return None

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
print("DFS path:", dfs(graph, 'A', 'F'))
```

---

## 2. Breadth First Search (BFS)

### 🔹 Concept:

Explores all neighbors at the present depth before going deeper.

### 🧠 Strategy:

FIFO (First In, First Out) – implemented using a queue.

### 💻 Python Code Example:

```python
from collections import deque

def bfs(graph, start, goal):
    visited = set()
    queue = deque([[start]])

    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None

print("BFS path:", bfs(graph, 'A', 'F'))
```

---

## 3. Uniform Cost Search (UCS)

### 🔹 Concept:

Expands the node with the lowest cumulative cost.

### 💰 Use Case:

When costs of paths vary.

### 💻 Python Code Example:

```python
import heapq

def ucs(graph, start, goal):
    queue = [(0, start, [])]
    visited = set()

    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        path = path + [node]
        visited.add(node)
        if node == goal:
            return (cost, path)
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path))
    return None

weighted_graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 6), ('E', 2)],
    'C': [('F', 3)],
    'D': [],
    'E': [('F', 1)],
    'F': []
}
print("UCS path and cost:", ucs(weighted_graph, 'A', 'F'))
```

---

## 4. Depth Limited Search (DLS)

### 🔹 Concept:

Depth First Search with a limit on how deep it can go.

### 💻 Python Code Example:

```python
def dls(graph, start, goal, limit):
    def recurse(node, path, depth):
        if depth > limit:
            return None
        path.append(node)
        if node == goal:
            return path
        for neighbor in graph[node]:
            result = recurse(neighbor, path.copy(), depth + 1)
            if result:
                return result
        return None

    return recurse(start, [], 0)

print("DLS path (limit=2):", dls(graph, 'A', 'F', 2))
```

---

## 5. Iterative Deepening Depth First Search (IDDFS)

### 🔹 Concept:

Combines DFS and BFS – uses DFS up to increasing depth limits.

### 💻 Python Code Example:

```python
def iddfs(graph, start, goal, max_depth):
    for depth in range(max_depth + 1):
        path = dls(graph, start, goal, depth)
        if path:
            return path
    return None

print("IDDFS path (max depth=5):", iddfs(graph, 'A', 'F', 5))
```

---

## 🧠 Summary

| Algorithm | Data Structure | Optimal | Complete | Use Case                      |
| --------- | -------------- | ------- | -------- | ----------------------------- |
| DFS       | Stack          | ❌      | ❌       | Large depths, low memory      |
| BFS       | Queue          | ✅      | ✅       | Shortest path in uniform cost |
| UCS       | Priority Queue | ✅      | ✅       | Varying path costs            |
| DLS       | Stack          | ❌      | ❌       | Depth-limited scenarios       |
| IDDFS     | Stack + Loop   | ✅      | ✅       | Combines benefits of DFS/BFS  |

# 🧠 Informed Search Algorithms & Constraint Satisfaction Problems

Informed search algorithms use **heuristic functions** to guide the search process, making them more efficient than uninformed (blind) methods.

We’ll explore:

- **Heuristic Functions**
- **Best First Search**
- **A\* Search**
- **Constraint Satisfaction Problems (CSP)**:
  - CryptoArithmetic (SEND + MORE = MONEY)
  - Map Coloring
  - N-Queens

---

## 🎯 Heuristic Functions

A **heuristic** is an estimate of the cost from the current node to the goal. A good heuristic improves search efficiency.

Example: In a grid, the **Manhattan distance** is often used as a heuristic.

```python
def manhattan_heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
```

---

## 🔹 Best First Search (Greedy)

Uses only the heuristic function to expand the node that seems closest to the goal.

### 💻 Python Code Example:

```python
import heapq

def best_first_search(graph, heuristics, start, goal):
    visited = set()
    queue = [(heuristics[start], start, [])]

    while queue:
        _, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == goal:
            return path
        for neighbor in graph[node]:
            heapq.heappush(queue, (heuristics[neighbor], neighbor, path))
    return None

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': ['F'], 'F': []
}

heuristics = {'A': 5, 'B': 3, 'C': 4, 'D': 6, 'E': 2, 'F': 0}

print("Best First Search path:", best_first_search(graph, heuristics, 'A', 'F'))
```

---

## ⭐ A\* Search

Uses both path cost and heuristic: `f(n) = g(n) + h(n)`.

### 💻 Python Code Example:

```python
def a_star(graph, heuristics, start, goal):
    import heapq
    queue = [(0 + heuristics[start], 0, start, [])]
    visited = set()

    while queue:
        f, cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == goal:
            return (cost, path)
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                g = cost + weight
                h = heuristics[neighbor]
                heapq.heappush(queue, (g + h, g, neighbor, path))
    return None

weighted_graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 6), ('E', 2)],
    'C': [('F', 3)],
    'D': [], 'E': [('F', 1)], 'F': []
}

heuristics = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 2, 'F': 0}

print("A* Search path and cost:", a_star(weighted_graph, heuristics, 'A', 'F'))
```

---

# 🎯 Constraint Satisfaction Problems (CSPs)

CSPs involve variables, domains, and constraints. Goal: Assign values to variables without violating constraints.

---

## 1. CryptoArithmetic: SEND + MORE = MONEY

```python
import itertools

def crypto_arithmetic():
    letters = 'SENDMORY'
    for perm in itertools.permutations(range(10), len(letters)):
        s, e, n, d, m, o, r, y = perm
        if s == 0 or m == 0:
            continue
        send = s*1000 + e*100 + n*10 + d
        more = m*1000 + o*100 + r*10 + e
        money = m*10000 + o*1000 + n*100 + e*10 + y
        if send + more == money:
            return {'SEND': send, 'MORE': more, 'MONEY': money}
    return None

print("CryptoArithmetic solution:", crypto_arithmetic())
```

---

## 2. Map Coloring

Coloring regions such that no adjacent regions share the same color.

```python
def is_valid_color(state, region, color, neighbors):
    for neighbor in neighbors[region]:
        if state.get(neighbor) == color:
            return False
    return True

def map_coloring(regions, colors, neighbors):
    def backtrack(state):
        if len(state) == len(regions):
            return state
        region = [r for r in regions if r not in state][0]
        for color in colors:
            if is_valid_color(state, region, color, neighbors):
                state[region] = color
                result = backtrack(state)
                if result:
                    return result
                del state[region]
        return None

    return backtrack({})

regions = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
colors = ['Red', 'Green', 'Blue']
neighbors = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'Q'],
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q': ['NT', 'SA', 'NSW'],
    'NSW': ['Q', 'SA', 'V'],
    'V': ['SA', 'NSW'],
    'T': []
}

print("Map Coloring solution:", map_coloring(regions, colors, neighbors))
```

---

## 3. N-Queens Problem

Place N queens on an N×N board such that no two queens threaten each other.

```python
def n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True

    def solve(board, row):
        if row == n:
            return [board.copy()]
        solutions = []
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solutions += solve(board, row + 1)
        return solutions

    return solve([-1] * n, 0)

print("One solution for 8-Queens:", n_queens(8)[0])
```

---

## ✅ Summary

| Technique  | Informed? | Uses Heuristics? | Example Application        |
| ---------- | --------- | ---------------- | -------------------------- |
| Best First | ✅        | ✅               | Shortest estimated path    |
| A\*        | ✅        | ✅               | Pathfinding (e.g., games)  |
| CSPs       | ✅        | 🚫 (but logical) | Puzzle solving, scheduling |

## 🎮 Adversarial Search in AI

Adversarial Search is used in scenarios where multiple agents (players) compete against each other, such as games like chess or tic-tac-toe. It models decision-making in **competitive environments**.

---

## 📌 Topics Covered

- Game Playing
- Min-Max Search
- Alpha-Beta Pruning

---

## ♟️ Game Playing in AI

A **game** can be represented as a tree where:

- **Nodes** = Game states
- **Edges** = Player moves
- **Terminal nodes** = End of the game (win/loss/draw)
- **Utility values** = Numerical scores assigned to end states

### 🧠 Game Characteristics

| Property              | Examples                           |
| --------------------- | ---------------------------------- |
| Turn-based            | Chess, Tic-Tac-Toe                 |
| Deterministic         | Chess, Checkers                    |
| Stochastic            | Backgammon (uses dice)             |
| Perfect Information   | Chess (both players see the board) |
| Imperfect Information | Poker (hidden cards)               |

---

## 🔁 Min-Max Algorithm

Minimax is used to **maximize your minimum gain**, assuming the opponent plays optimally.

### 💡 Key Concepts:

- **MAX** player: tries to maximize the score.
- **MIN** player: tries to minimize the opponent's score.

### 💻 Python Example: Simplified Minimax (Tic-Tac-Toe)

```python
def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner is not None:
        return winner

    if is_maximizing:
        best_score = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ''
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ''
                    best_score = min(score, best_score)
        return best_score

def check_winner(board):
    # Sample function (you'll need real logic)
    # Return +1 for X win, -1 for O win, 0 for draw, None otherwise
    return None
```

## ✂️ Alpha-Beta Pruning

Alpha-Beta Pruning improves Minimax by **ignoring branches** that cannot influence the final decision.

- **Alpha**: Best value that the maximizer can guarantee so far.
- **Beta**: Best value that the minimizer can guarantee so far.

---

### 💻 Python Example: With Alpha-Beta Pruning

```python
def alpha_beta(board, depth, alpha, beta, is_maximizing):
    winner = check_winner(board)
    if winner is not None:
        return winner

    if is_maximizing:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'X'
                    eval = alpha_beta(board, depth + 1, alpha, beta, False)
                    board[i][j] = ''
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'O'
                    eval = alpha_beta(board, depth + 1, alpha, beta, True)
                    board[i][j] = ''
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval
```

## ✅ Summary

| Concept             | Description                                    |
| ------------------- | ---------------------------------------------- |
| **Minimax**         | Decision rule for minimizing the possible loss |
| **Alpha-Beta**      | Cuts off branches in the game tree             |
| **Utility Value**   | Numerical outcome of a game                    |
| **MAX/MIN Players** | Competing agents                               |

---

## 📘 Use Cases

- Board games (chess, checkers, tic-tac-toe)
- Turn-based strategy games
- Any 2-player deterministic environment
