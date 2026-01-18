import numpy as np

GRAVITY = 9.81
INIT_ANGLE_MIN_MAX = 3

TRACK_Y = 0.0
L_DRAW = 1.0

# Action space: finer control near zero, coarser at extremes
ACTIONS = np.array([-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10])
N_ACTIONS = len(ACTIONS)

# State space discretization - SIMPLIFIED
X_BINS = 8
THETA_BINS = 12
X_DOT_BINS = 8
THETA_DOT_BINS = 12

X_RANGE = (-2.4, 2.4)
THETA_RANGE = (-0.5, 0.5)
X_DOT_RANGE = (-2.0, 2.0)
THETA_DOT_RANGE = (-2.0, 2.0)

# for drawing
TRACK_Y = 0.0
L_DRAW = 1.0

WIDTH, HEIGHT = 800, 400
SCALE = 100.0
CART_W, CART_H = 60, 30

# Q-learning parameters
ALPHA = 0.3        # learning rate (increased)
GAMMA = 0.95       # discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.015
EPSILON_DECAY = 0.999