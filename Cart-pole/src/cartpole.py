from config import *
import numpy as np

class CartPole:
    def __init__(self, cart_mass = 1.0, pole_mass = 0.1, pole_length=0.5, dt = 0.02):
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.dt = dt

        self.reset()

    def reset(self):
        initial_angle = np.random.uniform(-INIT_ANGLE_MIN_MAX, INIT_ANGLE_MIN_MAX)
        self.z = np.array([0.0, np.deg2rad(initial_angle), 0.0, 0.0])

    def discretize_state(self):
        x, theta, x_dot, theta_dot = self.z
        
        # Clip values to ranges
        x = np.clip(x, X_RANGE[0], X_RANGE[1])
        theta = np.clip(theta, THETA_RANGE[0], THETA_RANGE[1])
        x_dot = np.clip(x_dot, X_DOT_RANGE[0], X_DOT_RANGE[1])
        theta_dot = np.clip(theta_dot, THETA_DOT_RANGE[0], THETA_DOT_RANGE[1])
        
        # Convert to bin indices
        x_bin = min(int((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * X_BINS), X_BINS - 1)
        theta_bin = min(int((theta - THETA_RANGE[0]) / (THETA_RANGE[1] - THETA_RANGE[0]) * THETA_BINS), THETA_BINS - 1)
        x_dot_bin = min(int((x_dot - X_DOT_RANGE[0]) / (X_DOT_RANGE[1] - X_DOT_RANGE[0]) * X_DOT_BINS), X_DOT_BINS - 1)
        theta_dot_bin = min(int((theta_dot - THETA_DOT_RANGE[0]) / (THETA_DOT_RANGE[1] - THETA_DOT_RANGE[0]) * THETA_DOT_BINS), THETA_DOT_BINS - 1)
        
        return (max(0, x_bin), max(0, theta_bin), max(0, x_dot_bin), max(0, theta_dot_bin))

    def step(self, u):
        z_next = self.z + self.dt * self.dynamics(u)
        x, theta, _, _ = z_next

        cart_x = x
        cart_y = TRACK_Y
        pole_tip_x = cart_x + L_DRAW * np.sin(theta)
        pole_tip_y = cart_y + L_DRAW * np.cos(theta)

        cart_xy = (cart_x, cart_y)
        pole_xy = (pole_tip_x, pole_tip_y)
        return z_next, cart_xy, pole_xy

    def dynamics(self, u):
        x, theta, x_dot, theta_dot = self.z
        s, c = np.sin(theta), np.cos(theta)

        num_theta_dd = (self.cart_mass + self.pole_mass) * GRAVITY * s - c * (u + self.pole_mass * self.pole_length * theta_dot**2 * s)
        den_theta_dd = (1 + 1.0/3.0) * (self.cart_mass + self.pole_mass) * self.pole_length - self.pole_mass * self.pole_length * c**2
        theta_dd = num_theta_dd / den_theta_dd

        x_dd = (self.pole_mass * GRAVITY * s * c - (1 + 1.0/3.0) * (u + self.pole_mass * self.pole_length * theta_dot**2 * s)) / (self.pole_mass * c**2 - (1 + 1.0/3.0) * (self.cart_mass + self.pole_mass))

        return np.array([x_dot, theta_dot, x_dd, theta_dd])

    def is_terminal(self):
        """Check if state is terminal"""
        x, theta, _, _ = self.z
        return abs(theta) > 0.5 or abs(x) > 2.4
    
    def get_reward(self, z_next, terminated):
        if terminated:
            return -200000  # Large penalty for failure
        
        # Reward for staying upright
        theta = z_next[1]
        x = z_next[0]
        theta_dot = z_next[3]
        
        # Primary reward: small angle is good
        angle_reward = 1.0 - (abs(theta) / 0.5) ** 2
        
        # Bonus for low angular velocity (stable)
        stability_reward = 0.5 * (1.0 - min(abs(theta_dot) / 2.0, 1.0))
        
        # Small penalty for being far from center
        center_penalty = -0.1 * (abs(x) / 2.4) ** 2
        
        return angle_reward + stability_reward + center_penalty