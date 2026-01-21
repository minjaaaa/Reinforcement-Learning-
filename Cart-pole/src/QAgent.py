from config import *
import numpy as np

class QLearningAgent:
    def __init__(self):
        # Initialize Q-table with small random values for exploration
        self.q_table = np.random.uniform(-0.01, 0.01, (X_BINS, THETA_BINS, X_DOT_BINS, THETA_DOT_BINS, N_ACTIONS))
        self.epsilon = EPSILON_START
        
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, terminated):
        """Q-learning update"""
        current_q = self.q_table[state][action]
        
        if terminated:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + GAMMA * max_next_q
        
        # Update Q-value
        self.q_table[state][action] += ALPHA * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)