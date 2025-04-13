# Knowledge-Based Agent

A Knowledge-Based Agent is an intelligent agent that uses a knowledge base (KB) to make decisions. It perceives its environment, reasons about it using stored knowledge, and acts to achieve goals.

## Key Components

- **Knowledge Base (KB)**: A set of sentences (facts and rules) expressed in a formal language.
- **Inference Engine**: Mechanisms to derive new knowledge from the KB.
- **Perception**: Input from sensors to update the KB.
- **Action**: Output to actuators based on reasoning.

## How It Works

1. **Perceive**: The agent receives input from the environment.
2. **Update KB**: Incorporates new percepts into the KB.
3. **Reason**: Uses inference to query what action to take.
4. **Act**: Executes the action.

## Visualization

Imagine a robot vacuum:
- **Percept**: Detects dirt (sensor input).
- **KB**: "If dirt is detected, vacuum."
- **Inference**: Concludes it should vacuum.
- **Action**: Activates vacuum motor.

## Code Example (Python - Simplified KB Agent)

```python
class KnowledgeBasedAgent:
    def __init__(self):
        self.kb = []  # Knowledge base as a list of facts/rules

    def tell(self, fact):
        """Add fact to KB."""
        self.kb.append(fact)

    def ask(self, query):
        """Check if query is true based on KB."""
        return query in self.kb

    def act(self, percept):
        """Decide action based on percept and KB."""
        self.tell(percept)  # Update KB with new percept
        if self.ask("dirt_detected"):
            return "vacuum"
        return "do_nothing"

# Example usage
agent = KnowledgeBasedAgent()
print(agent.act("dirt_detected"))  # Output: vacuum
print(agent.act("no_dirt"))        # Output: do_nothing
```

# Overview of Propositional Logic

Propositional Logic is a formal system for reasoning where statements (propositions) are either true or false. It uses symbols and connectives to represent and manipulate knowledge.

## Key Concepts

- **Propositions**: Declarative statements (e.g., "It is raining" = P).
- **Connectives**:
  - **AND (∧)**: P ∧ Q is true if both P and Q are true.
  - **OR (∨)**: P ∨ Q is true if at least one is true.
  - **NOT (¬)**: ¬P is true if P is false.
  - **IMPLIES (→)**: P → Q is false only if P is true and Q is false.
  - **BICONDITIONAL (↔)**: P ↔ Q is true if P and Q have the same truth value.
- **Truth Tables**: Used to evaluate expressions.

## Truth Table Example (P ∧ Q)

| P     | Q     | P ∧ Q |
|-------|-------|-------|
| True  | True  | True  |
| True  | False | False |
| False | True  | False |
| False | False | False |

## Code Example (Python - Truth Table Generator)

```python
def and_connective(p, q):
    return p and q

def print_truth_table():
    print("| P     | Q     | P ∧ Q |")
    print("|-------|-------|-------|")
    for p in [True, False]:
        for q in [True, False]:
            result = and_connective(p, q)
            print(f"| {p!s:<5} | {q!s:<5} | {result!s:<5} |")

print_truth_table()
```

# First-Order Predicate Logic

First-Order Predicate Logic (FOPL) extends propositional logic by introducing predicates, quantifiers, and variables to represent relationships and properties in a more expressive way.

## Key Concepts

- **Predicates**: Functions that return true/false (e.g., Loves(John, Mary)).
- **Constants**: Specific objects (e.g., John, Mary).
- **Variables**: Placeholders (e.g., x, y).
- **Quantifiers**:
  - **Universal (∀)**: "For all" (e.g., ∀x Person(x) → Mortal(x)).
  - **Existential (∃)**: "There exists" (e.g., ∃x Loves(x, Mary)).
- **Connectives**: Same as propositional logic.

## Inference in FOPL

Inference derives new sentences from existing ones. Key methods include:

### 1. Forward Chaining
**Definition**: Starts with known facts and applies rules to derive new facts until the goal is reached.

**Example**:
- KB: ∀x Mortal(x) → Dies(x), Mortal(Socrates)
- Derive: Dies(Socrates)

**Process**:
- Match premises of rules to KB facts.
- Add conclusions to KB.
- Repeat until goal or no new facts.

### 2. Backward Chaining
**Definition**: Starts with the goal and works backward to find supporting facts.

**Example**:
- Goal: Dies(Socrates)
- KB: ∀x Mortal(x) → Dies(x), Mortal(Socrates)
- Check: Mortal(Socrates) is true, so Dies(Socrates) is true.

**Process**:
- Select goal.
- Find rules/conclusions that imply the goal.
- Recursively prove premises.

### 3. Resolution
**Definition**: A proof method that converts sentences to Conjunctive Normal Form (CNF) and resolves contradictions.

**Steps**:
- Convert KB and negated goal to CNF.
- Apply resolution rule: If P ∨ Q and ¬P ∨ R, derive Q ∨ R.
- Continue until a contradiction (empty clause) is found or no progress is made.

**Example**:
- KB: Mortal(Socrates) → Dies(Socrates), Mortal(Socrates)
- Goal: Dies(Socrates)
- Negated goal: ¬Dies(Socrates)
- Resolve to derive contradiction.

## Code Example (Python - Simplified Forward Chaining)

```python
class KnowledgeBase:
    def __init__(self):
        self.facts = set()
        self.rules = []  # List of (premise, conclusion)

    def tell_fact(self, fact):
        self.facts.add(fact)

    def tell_rule(self, premise, conclusion):
        self.rules.append((premise, conclusion))

    def forward_chain(self, goal):
        while True:
            new_facts = False
            for premise, conclusion in self.rules:
                if premise in self.facts and conclusion not in self.facts:
                    self.facts.add(conclusion)
                    new_facts = True
                    if conclusion == goal:
                        return True
            if not new_facts:
                return False

# Example
kb = KnowledgeBase()
kb.tell_fact("Mortal(Socrates)")
kb.tell_rule("Mortal(Socrates)", "Dies(Socrates)")
print(kb.forward_chain("Dies(Socrates)"))  # Output: True
```

# Planning

Planning involves generating a sequence of actions to achieve a goal state from an initial state. It's widely used in AI for tasks like robot navigation or scheduling.

## 1. Planning with State-Space Search

**Definition**: Models planning as a search problem in a state-space graph.

**Components**:
- States: Possible world configurations.
- Actions: Transitions between states.
- Goal Test: Checks if a state satisfies the goal.
- Path Cost: Cost of action sequences.

**Search Algorithms**:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* (heuristic-based)

**Example**: Robot moving from Room A to Room B.
- Initial State: Robot in A.
- Goal State: Robot in B.
- Actions: Move(A, B).

## Code Example (Python - Simple State-Space Search with BFS)

```python
from collections import deque

def bfs(initial_state, goal_state, actions):
    queue = deque([(initial_state, [])])
    visited = set()

    while queue:
        state, path = queue.popleft()
        if state == goal_state:
            return path
        if state in visited:
            continue
        visited.add(state)
        for action, next_state in actions(state):
            queue.append((next_state, path + [action]))
    return None

# Example: Robot navigation
def actions(state):
    transitions = {
        "A": [("move_to_B", "B")],
        "B": [("move_to_A", "A")]
    }
    return transitions.get(state, [])

print(bfs("A", "B", actions))  # Output: ['move_to_B']
```

## 2. Partial-Order Planning (POP)

**Definition**: Plans by maintaining a partial order of actions instead of a strict sequence, allowing flexibility.

**Key Ideas**:
- Plan: Set of actions and ordering constraints (e.g., Action A before B).
- Causal Links: Ensure actions achieve preconditions (e.g., A achieves x for B).
- Threats: Resolve conflicts where an action might undo a condition.

**Algorithm**:
1. Start with initial and goal states.
2. Select open preconditions.
3. Add actions or reuse existing ones to satisfy preconditions.
4. Resolve threats by adding ordering constraints.
5. Repeat until no open preconditions.

## Code Example (Python - Simplified POP)

```python
class PartialOrderPlan:
    def __init__(self, initial, goal):
        self.actions = []
        self.orderings = []  # (action1, action2) means action1 before action2
        self.open_preconditions = [(goal, None)]  # (condition, action needing it)

    def add_action(self, action, preconditions, effects):
        self.actions.append(action)
        for effect in effects:
            for cond, needing_action in self.open_preconditions[:]:
                if cond == effect:
                    self.orderings.append((action, needing_action))
                    self.open_preconditions.remove((cond, needing_action))
        for precond in preconditions:
            self.open_preconditions.append((precond, action))

# Example
plan = PartialOrderPlan("at_A", "at_B")
plan.add_action("move_to_B", ["at_A"], ["at_B"])
print(plan.orderings)  # Output: [('move_to_B', None)]
```

## 3. Hierarchical Planning

**Definition**: Breaks planning into high-level and low-level tasks, refining abstract plans into detailed ones.

**Key Ideas**:
- Abstract Actions: High-level goals (e.g., "Travel to B").
- Refinement: Decompose into primitive actions (e.g., "Walk to car", "Drive").
- Hierarchical Task Network (HTN): Common framework.

**Example**:
- High-level: "Prepare dinner."
- Low-level: "Chop vegetables", "Cook pasta."

## Code Example (Python - Simplified HTN)

```python
class HTNPlanner:
    def __init__(self):
        self.tasks = {}  # task -> subtasks

    def add_task(self, task, subtasks):
        self.tasks[task] = subtasks

    def plan(self, task):
        if task not in self.tasks:
            return [task]  # Primitive task
        plan = []
        for subtask in self.tasks[task]:
            plan.extend(self.plan(subtask))
        return plan

# Example
planner = HTNPlanner()
planner.add_task("prepare_dinner", ["chop_vegetables", "cook_pasta"])
print(planner.plan("prepare_dinner"))  # Output: ['chop_vegetables', 'cook_pasta']
```

## 4. Conditional Planning

**Definition**: Plans for uncertain environments by including conditional actions (e.g., "If X, do A; else, do B").

**Key Ideas**:
- Contingency Plans: Handle different outcomes.
- Sensing Actions: Gather information during execution.

**Example**:
- Goal: Get to B.
- Plan: "Check if door is open; if yes, walk; if no, unlock then walk."

## Code Example (Python - Simplified Conditional Plan)

```python
class ConditionalPlanner:
    def __init__(self):
        self.plans = {}

    def add_conditional_plan(self, condition, action):
        self.plans[condition] = action

    def execute(self, condition):
        return self.plans.get(condition, "do_nothing")

# Example
planner = ConditionalPlanner()
planner.add_conditional_plan("door_open", "walk")
planner.add_conditional_plan("door_locked", "unlock_then_walk")
print(planner.execute("door_open"))  # Output: walk
```
