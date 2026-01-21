"""
Value Iteration Agent for Tic-Tac-Toe
Implements reinforcement learning using the Value Iteration algorithm
"""

import pickle
import json
from typing import Dict, List, Tuple, Optional
from game_state import TicTacToeGame
from typing import Optional

class ValueIterationAgent:
    """
    Value Iteration Agent for Tic-Tac-Toe
    
    Uses Bellman equation to compute optimal value function V(s) and policy π(s)
    V(s) ← max_a Σ P(s'|s,a)[R(s') + γV(s')]
    """
    
    def __init__(self, 
                 opponent,
                 agent_symbol: str = 'X',
                 gamma: float = 0.9,
                 theta: float = 0.001):
        """
        Initialize Value Iteration Agent
        
        Args:
            opponent: Opponent policy object (RandomOpponent or HeuristicOpponent)
            agent_symbol: 'X' or 'O' - the symbol this agent plays
            gamma: Discount factor (0 < γ < 1)
            theta: Convergence threshold
        """
        self.opponent = opponent
        self.agent_symbol = agent_symbol
        self.opponent_symbol = 'X' if agent_symbol == 'O' else 'O'
        self.gamma = gamma
        self.theta = theta
        
        # Set opponent's symbol
        self.opponent.symbol = self.opponent_symbol
        if hasattr(self.opponent, 'opponent_symbol'):
            self.opponent.opponent_symbol = self.agent_symbol
        
        # Game instance
        self.game = TicTacToeGame()
        
        # Value function and policy
        self.V: Dict[str, float] = {}
        self.policy: Dict[str, int] = {}
        
        # All valid states
        self.all_states = set()
        
    def train(self, max_iterations: int = 1000) -> Tuple[bool, int]:
        """
        Run value iteration until convergence
        
        Algorithm:
        1. Initialize V(s) = 0 for all states
        2. Repeat:
           a. For each non-terminal state:
              - Compute Q(s,a) for all actions
              - Update V(s) = max_a Q(s,a)
           b. Check convergence
        3. Extract optimal policy
        
        Args:
            max_iterations: Maximum number of iterations
        
        Returns:
            Tuple of (converged: bool, iterations: int)
        """
        print("\nTraining Value Iteration Agent")
        print(f"Opponent: {self.opponent.__class__.__name__}")
        print(f"Agent Role: {self.agent_symbol} ({'First' if self.agent_symbol == 'X' else 'Second'} Player)")
        print(f"Gamma: {self.gamma}, Theta: {self.theta}")
        print()
        
        # Get all valid states
        print("Generating all valid board states...")
        self.all_states = self.game.get_all_states()
        print(f"Total states: {len(self.all_states)}")
        
        # Filter states where it's agent's turn
        agent_states = self._filter_agent_states()
        print(f"States where agent moves: {len(agent_states)}")
        
        # Initialize V(s) = 0 for all states
        for state in self.all_states:
            self.V[state] = 0.0
        
        # Value iteration loop
        converged = False
        iteration = 0
        
        for iteration in range(1, max_iterations + 1):
            delta = 0.0
            
            # Update value for each state where agent can move
            for state in agent_states:
                if self.game.is_terminal(state):
                    continue
                
                old_value = self.V[state]
                
                # Compute Q(s,a) for all actions and take max
                max_q_value = float('-inf')
                available_actions = self.game.get_available_moves(state)
                
                if not available_actions:
                    continue
                
                for action in available_actions:
                    q_value = self._compute_q_value(state, action)
                    max_q_value = max(max_q_value, q_value)
                
                # Update value
                self.V[state] = max_q_value
                
                # Track maximum change
                delta = max(delta, abs(old_value - self.V[state]))
            
            # Print progress every 10 iterations
            if iteration % 10 == 0 or iteration == 1:
                print(f"Iteration {iteration}: Max delta = {delta:.6f}")
            
            # Check convergence
            if delta < self.theta:
                converged = True
                print(f"\nConverged after {iteration} iterations")
                break
        
        if not converged:
            print(f"\nDid not converge after {max_iterations} iterations")
        
        # Extract optimal policy
        print("Extracting optimal policy...")
        self._extract_policy()
        print(f"Policy extracted for {len(self.policy)} states")
        
        return converged, iteration
    
    def _filter_agent_states(self) -> List[str]:
        """
        Filter states where it's the agent's turn to move
        
        Returns:
            List of states where agent should move
        """
        agent_states = []
        
        for state in self.all_states:
            # Skip terminal states
            if self.game.is_terminal(state):
                continue
            
            # Determine whose turn it is
            x_count = state.count('X')
            o_count = state.count('O')
            
            # X goes first
            current_turn = 'X' if x_count == o_count else 'O'
            
            # Only include states where it's agent's turn
            if current_turn == self.agent_symbol:
                agent_states.append(state)
        
        return agent_states
    
    def _compute_q_value(self, state: str, action: int) -> float:
        """
        Compute Q(s,a) = Σ P(s'|s,a)[R(s') + γV(s')]
        
        Args:
            state: Current board state
            action: Action to take (position 0-8)
        
        Returns:
            Q-value for state-action pair
        """
        # Get all possible next states and their probabilities
        next_states = self._get_next_states(state, action)
        
        q_value = 0.0
        
        for next_state, probability in next_states:
            # Get reward for next state
            reward = self.game.get_reward(next_state, self.agent_symbol)
            
            # Add to Q-value: P(s'|s,a) * [R(s') + γV(s')]
            q_value += probability * (reward + self.gamma * self.V.get(next_state, 0.0))
        
        return q_value
    
    def _get_next_states(self, state: str, action: int) -> List[Tuple[str, float]]:
        """
        Get all possible next states after agent action and opponent response
        
        Process:
        1. Apply agent's action → intermediate state
        2. If terminal, return [(intermediate_state, 1.0)]
        3. Simulate opponent's moves with probabilities
        
        Args:
            state: Current board state
            action: Agent's action (position 0-8)
        
        Returns:
            List of (next_state, probability) tuples
        """
        # Apply agent's move
        intermediate_state = self.game.apply_move(state, action, self.agent_symbol)
        
        # If game ended, return this state with probability 1.0
        if self.game.is_terminal(intermediate_state):
            return [(intermediate_state, 1.0)]
        
        # Get opponent's possible moves
        opponent_moves = self.game.get_available_moves(intermediate_state)
        
        if not opponent_moves:
            return [(intermediate_state, 1.0)]
        
        # Determine probabilities based on opponent type
        next_states = []
        
        # Check if opponent is deterministic (has fixed policy)
        if hasattr(self.opponent, '_find_winning_move'):
            # Deterministic opponent (Heuristic)
            opponent_move = self.opponent.get_move(intermediate_state)
            
            if opponent_move is not None:
                next_state = self.game.apply_move(intermediate_state, opponent_move, self.opponent_symbol)
                next_states.append((next_state, 1.0))
            else:
                next_states.append((intermediate_state, 1.0))
        else:
            # Random opponent - uniform probability
            prob = 1.0 / len(opponent_moves)
            
            for opp_move in opponent_moves:
                next_state = self.game.apply_move(intermediate_state, opp_move, self.opponent_symbol)
                next_states.append((next_state, prob))
        
        return next_states
    
    def _extract_policy(self):
        """
        Extract optimal policy π(s) = arg max_a Q(s,a)
        
        For each non-terminal state where agent moves,
        select the action with highest Q-value
        """
        self.policy = {}
        
        agent_states = self._filter_agent_states()
        
        for state in agent_states:
            if self.game.is_terminal(state):
                continue
            
            available_actions = self.game.get_available_moves(state)
            
            if not available_actions:
                continue
            
            # Find action with maximum Q-value
            best_action = None
            best_q_value = float('-inf')
            
            for action in available_actions:
                q_value = self._compute_q_value(state, action)
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            if best_action is not None:
                self.policy[state] = best_action
    
    def get_move(self, board: str, agent_symbol: Optional[str] = None) -> Optional[int]:
        """
        Use learned policy to select move
        
        Args:
            board: 9-character string representing board state
            agent_symbol: Optional symbol (uses self.agent_symbol if not provided)
        
        Returns:
            Position (0-8) to place symbol, or None if no policy exists
        """
        if agent_symbol is None:
            agent_symbol = self.agent_symbol
        
        # Check if policy exists for this state
        if board in self.policy:
            return self.policy[board]
        
        # Fallback: choose first available move
        available = self.game.get_available_moves(board)
        return available[0] if available else None
    
    def save_policy(self, filename: str):
        """
        Save learned value function and policy to file
        
        Args:
            filename: Path to save file (will use .pkl extension)
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        data = {
            'V': self.V,
            'policy': self.policy,
            'agent_symbol': self.agent_symbol,
            'opponent_symbol': self.opponent_symbol,
            'gamma': self.gamma,
            'theta': self.theta,
            'opponent_type': self.opponent.__class__.__name__
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nPolicy saved to: {filename}")
    
    def load_policy(self, filename: str):
        """
        Load previously learned policy
        
        Args:
            filename: Path to saved policy file
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.V = data['V']
        self.policy = data['policy']
        self.agent_symbol = data['agent_symbol']
        self.opponent_symbol = data['opponent_symbol']
        self.gamma = data['gamma']
        self.theta = data['theta']
        
        print(f"Policy loaded from: {filename}")
        print(f"States in value function: {len(self.V)}")
        print(f"States in policy: {len(self.policy)}")
    
    def get_state_value(self, board: str) -> float:
        """
        Get value V(s) for given board state
        
        Args:
            board: 9-character string representing board state
        
        Returns:
            Value of the state
        """
        return self.V.get(board, 0.0)


# Test the implementation
if __name__ == "__main__":
    from opponent_policies import RandomOpponent, HeuristicOpponent
    
    print("Testing Value Iteration Agent")
    print("=" * 60)
    
    # Test with random opponent
    print("\nTraining against Random Opponent (Agent as X)")
    random_opp = RandomOpponent('O')
    agent = ValueIterationAgent(random_opp, agent_symbol='X', gamma=0.9, theta=0.001)
    
    converged, iterations = agent.train(max_iterations=100)
    
    print(f"\nTraining complete!")
    print(f"Converged: {converged}")
    print(f"Iterations: {iterations}")
    
    # Test policy on empty board
    empty_board = '.' * 9
    move = agent.get_move(empty_board)
    print(f"\nAgent's first move on empty board: {move}")
    print(f"State value: {agent.get_state_value(empty_board):.4f}")
    
    # Save policy
    agent.save_policy('test_agent_random_X.pkl')
