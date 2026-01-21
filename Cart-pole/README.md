# Cart-Pole Balancing with Q-Learning

Classic reinforcement learning problem solved using tabular Q-Learning with real-time Pygame visualization.

## Problem Description

Balance a pole attached to a moving cart by applying left/right forces. The agent learns through trial and error to keep the pole upright.

**State Space (4D):**
- Cart position: [-2.4, 2.4]
- Pole angle: [-0.5, 0.5] radians (~±28.6°)
- Cart velocity: [-2.0, 2.0]
- Angular velocity: [-2.0, 2.0]

**Actions:** 11 discrete forces from -10N to +10N

**Goal:** Maximize steps before pole falls or cart leaves track

## Project Structure

```
Cart-pole/
├── README.md
└── src/
    ├── main.py       # Training loop & visualization
    ├── cartpole.py   # Physics environment
    ├── QAgent.py     # Q-Learning agent
    └── config.py     # Hyperparameters
```

## Algorithm: Q-Learning

**Q-Learning update rule:**
```
Q(s,a) ← Q(s,a) + α[R + γ max Q(s',a') - Q(s,a)]
```

**Key hyperparameters:**
- Learning rate (α): 0.3
- Discount factor (γ): 0.95
- Epsilon decay: 1.0 → 0.01

**State discretization:** 8×12×8×12 = 9,216 states

## Features

- **Real-time visualization** with Pygame
- **Shaped rewards** for faster learning (angle, stability, centering)
- **Interactive controls** during training
- **Save/Load** trained Q-tables
- **Performance tracking** (best, average over 100 episodes)

## Usage

### Installation
```bash
pip install numpy pygame
```

### Run Training
```bash
python main.py
```

### Controls
| Key | Action |
|-----|--------|
| `T` | Toggle training ON/OFF |
| `S` | Save Q-table |
| `L` | Load Q-table |
| `F` | Fast mode (no rendering) |
| `R` | Reset agent |

## Training Progress

- **Episodes 0-100:** Quick failures (<30 steps)
- **Episodes 100-500:** Improvement (30-200 steps)
- **Episodes 500+:** Mastery (200-500+ steps)

Typical training: 500-1000 episodes (~10-20 minutes)

## Reward Function

```
Terminal state: -200,000
Otherwise:
  - Angle reward: 1.0 - (|θ|/0.5)²
  - Stability: 0.5 × (1 - min(|θ̇|/2.0, 1))
  - Center penalty: -0.1 × (|x|/2.4)²
```

Encourages upright pole, low angular velocity, and staying centered.

## Implementation Highlights

- **Physics:** Lagrangian mechanics with Euler integration (dt=0.02s)
- **Exploration:** ε-greedy with exponential decay
- **Memory efficient:** ~800KB Q-table
- **Modular design:** Easy to extend or modify

## Results

Well-trained agent achieves:
- 20000+ steps consistently
- Stable balancing behavior