# 🧩 Maze Reinforcement Learning Project

Grid-based maze environment implementing classic **Reinforcement Learning (RL)** algorithms with rich **interactive visualization** using Matplotlib.

---

## 📌 Project Overview

* **Environment**: Grid-based Maze (MDP)
* **Algorithms**:

  * Value Iteration
  * Policy Iteration
  * Greedy Policies
* **Visualization**:

  * Interactive Matplotlib board
  * State values, Q-values, policy arrows
  * Animated agent simulation

---

## 📁 Project Structure

```
Maze/
├── README.md
└── src/
    ├── maze.py         # Main RL algorithms and environment logic
    ├── board.py        # MazeBoard class for visualization and rendering
    ├── cells.py        # Cell types and state representation
    └── __pycache__/    # Python bytecode cache
```

---

## 🧠 Core Components

### 1️⃣ `cells.py` – State Representation

Defines the maze cell hierarchy and state abstraction.

#### 🔹 Position

* Immutable `@dataclass`
* Attributes: `row`, `col`
* Callable → returns `(row, col)`

#### 🔹 Actions (Enum)

```text
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
```

#### 🔹 Cell (Abstract Base Class)

**Properties**:

* `reward: float`
* `steppable: bool`
* `terminal: bool`
* `teleport: bool`

**Methods**:

* `get_reward()`
* `is_steppable()`
* `is_terminal()`
* `is_teleport()`
* `get_next_cell()`

#### 🔹 Cell Types

| Cell Type    | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| **RegCell**  | Regular cell (`-1`) or penalty cell (`-10`), color: white/red |
| **TermCell** | Terminal (goal) cell, reward `0`, color: yellow               |
| **WallCell** | Obstacle, non-steppable, color: black                         |
| **TelCell**  | Teleport cell with destination position, color: blue          |

---

### 2️⃣ `board.py` – Environment Visualization

Handles maze rendering and interaction via **Matplotlib**.

#### 🔹 Initialization

* Random grid generation:

  * 67% regular cells (`-1`)
  * 13% penalty cells (`-10`)
  * 13% walls
  * 7% teleports
* One randomly placed terminal cell
* Mouse click event handling

#### 🔹 Visualization Features

* Color-coded grid
* State values `V(s)`
* Policy arrows (↑ ↓ ← →)
* Full Q-value display per cell
* Animated agent movement
* Teleport index mapping

#### 🔹 Main Methods

* `draw_board()`
* `draw_values(values)`
* `draw_actions(policy)`
* `draw_q_values(q_values)`
* `draw_agent(position, symbol)`
* `onclick(event)`

---

### 3️⃣ `maze.py` – RL Algorithms

Contains the **MazeEnvironment** and core RL logic.

#### 🔹 MazeEnvironment

* Wraps `MazeBoard`
* Extracts valid (steppable) states
* Handles teleports and rewards
* Defines state transitions

**Key Methods**:

* `init_states()`
* `init_actions()`
* `next_state(s, a)`
* `update_state_value()`
* `update_all_state_values()`
* `update_action_value()`
* `update_all_action_values()`

---

## 🤖 Implemented Algorithms

### A) Value Iteration

* Uses Bellman optimality equation:

```
V(s) = maxₐ [ R(s,a) + γ · V(s') ]
```

* Iterates until convergence (`error < ε`)
* Produces optimal value function **V***

---

### B) Policy Iteration

1. **Policy Evaluation** → Compute `V^π(s)`
2. **Policy Improvement** → Update `π(s)` via greedy Q

* Repeats until policy stabilizes
* Often faster than Value Iteration

---

### C) Greedy Policies

* `greedy()` → based on `V(s)`
* `greedy_q()` → based on `Q(s,a)`

Used for policy improvement and execution.

---

### D) Policy Execution

* `apply_policy()` simulates one episode
* Returns total discounted reward (gain)
* Animates agent movement on the board

---

## 🔁 Basic Workflow

### Step 1: Environment Setup

```python
board = MazeBoard(rows=10, cols=10)
env = MazeEnvironment(board)
```

### Step 2: Initialize Values & Policy

```python
v = {s: 0 for s in env.states}
policy = {s: random.choice(list(Actions)).name for s in env.states}
```

### Step 3: Value Iteration

```python
v_optimal = value_iteration(
    update=env.update_all_state_values,
    values=v,
    gamma=0.9,
    eps=0.01,
    iterations=100
)
```

### Step 4: Extract Greedy Policy

```python
optimal_policy = {
    s: greedy(env, s, v_optimal, gamma=0.9).name
    for s in env.states
}
```

### Step 5: Visualization & Simulation

```python
board.draw_board()
board.draw_values(v_optimal)
board.draw_actions(optimal_policy)

gain = apply_policy(env, greedy, start_state, gamma, v_optimal)
```

---

## 📘 Reinforcement Learning Concepts

### 🔹 Markov Decision Process (MDP)

* **States**: Grid positions
* **Actions**: {UP, DOWN, LEFT, RIGHT}
* **Transitions**: Deterministic
* **Rewards**: Cell-based
* **γ (Discount factor)**

---

### 🔹 Bellman Equations

**Optimality (VI)**:

```
V*(s) = maxₐ [ R(s,a) + γ · V*(s') ]
```

**Policy Evaluation (PI)**:

```
V^π(s) = Σ π(a|s) [ R(s,a) + γ · V^π(s') ]
```

---

### 🔹 Convergence

* Tracks max value change
* Stops when:

```
max |V_new - V_old| < ε
```

* Guaranteed for `γ < 1`

---

## ✨ Special Features

### 🔹 Teleportation

* Teleport cells redirect agent instantly
* Reward equals destination cell reward
* Invalid teleports auto-corrected
* Visualized with blue cells + index labels

### 🔹 Interactive UI

* Mouse click inspection
* Live Q-value rendering
* Smooth agent animation
* Clear color-coded grid

---

## ⚙️ Key Parameters

### Discount Factor (γ)

* `0.9 – 0.99`: long-term planning
* `0.1 – 0.5`: short-term rewards

### Convergence Threshold (ε)

* Smaller → more precise, slower
* Larger → faster, less accurate

### Iterations

* Typical range: `50 – 200`
* Safety cap for convergence


