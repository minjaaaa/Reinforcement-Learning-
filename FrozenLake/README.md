# 🧊 FrozenLake Reinforcement Learning Project

Grid-based FrozenLake environment implementing classic Reinforcement Learning (RL) algorithms with detailed visualization, policy inspection, and episode simulation.

---

##  Project Overview

**Environment:**  
FrozenLake (Grid World MDP, OpenAI Gymnasium compatible)

**Algorithms Implemented:**
- Monte Carlo Control
  - Naive Monte Carlo (Every-Visit)
  - Incremental Monte Carlo
- ε-greedy policy learning
- Greedy policy extraction from Q-values

**Visualization & Analysis:**
- Matplotlib-based grid visualization
- Policy arrows (← ↓ → ↑)
- Interactive Q-value inspection per state
- Learning curves (rolling average reward)
- Episode simulation with learned policy

---

## Project Structure

```
FrozenLake/
├── README.md
└── src/
    ├── Incremental_MC.py         # Implementation of Incremental MC method 
    └── Naive_MC.py               # Implementation of Naive MC method
```


---

## Core Components

### 1️⃣ Environment Setup

The project uses **Gymnasium's FrozenLake-v1** environment.

**Key properties:**
- Grid sizes: 8×8 (works for 4x4)
- Stohastic Dynamics (works with Deterministic)
- Terminal states: Goal (G) and Holes (H)
- Actions:
  - `0`: LEFT
  - `1`: DOWN
  - `2`: RIGHT
  - `3`: UP

```python
env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True # Adds Randomness
)
```
### 2️⃣ State & Action Representation

- **States:** integer indices (flattened grid)
- **Actions:** discrete set `{LEFT, DOWN, RIGHT, UP}`

**Q-table:**


- Q[s, a] → action-value estimate


---

## Implemented Algorithms

### A) Naive Monte Carlo Control

Uses full return history for each state–action pair.

**Characteristics:**
- Stores all observed returns
- Low bias, high variance
- Memory intensive
- Conceptually closest to the theoretical Monte Carlo definition

---

### B) Incremental Monte Carlo Control

Updates Q-values online without storing return history.


**Characteristics:**
- Memory efficient
- Numerically stable
- Equivalent to naive MC with incremental averaging

---

### C) ε-Greedy Policy Learning

During episode generation:

- ε decays gradually
- Ensures sufficient exploration
- Required for Monte Carlo control convergence

---

## Visualization

### Policy Visualization

Color-coded grid:
- **Start (S):** Blue  
- **Frozen (F):** White  
- **Hole (H):** Red  
- **Goal (G):** Green  

Arrows indicate the greedy action per state.

```python
plot_policy_on_map(Q, env)
```
### Learning Curve

- Rolling average reward
- Visualizes convergence behavior
- Useful for comparing algorithms

```python
plt.plot(rolling_average_rewards)
```
## Basic Workflow

### Step 1: Initialize Environment

```python
env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name='8x8',
        is_slippery=slippery,
        render_mode='human' if render else None,
        success_rate= 8.0 / 10.0,
        reward_schedule=(1, 0, 0)
    )
```
### Step 2: Initialize Q-table

```python
Q = np.zeros(
    (env.observation_space.n, env.action_space.n)
)
```

### Step 3: Monte Carlo Training

```python
start(episodes=100000, render=True, slippery=True)
```

### Step 4: Visualize Learned Policy

```python
plot_policy_with_q_inspector(Q, env)
```

### Step 5: Run Simulation

```python
simulate_episode(Q, env)
```

## Parameters
### Discount Factor (γ)

* 0.9 – 0.99: long-term planning

* 0.5 – 0.7: short-term rewards

### Exploration Rate (ε)

* Starts high (≈ 1.0)

* Decays slowly

* Lower bound > 0

### Episodes

* Typical range: 50k – 200k

* Required due to high Monte Carlo variance

### Special Features

* Randomized start states (optional)

* Deterministic vs stochastic dynamics

* Clean separation of learning & visualization

* Fully compatible with Gymnasium

## Notes

* Monte Carlo methods require complete episodes

* High variance is expected in FrozenLake

* Deterministic environments converge faster

* Random start states change policy interpretation
