"""
Opponent Policy Implementations for Tic-Tac-Toe
Includes Random and Heuristic (rule-based) opponents
"""

import random
from typing import Optional, List


class RandomOpponent:
    """
    Opponent that chooses moves uniformly at random
    """
    
    def __init__(self, symbol: str = 'O'):
        """
        Initialize random opponent
        
        Args:
            symbol: 'X' or 'O' - the symbol this opponent plays
        """
        self.symbol = symbol
        self.opponent_symbol = 'X' if symbol == 'O' else 'O'
    
    def get_move(self, board: str) -> Optional[int]:
        """
        Select a random available move
        
        Args:
            board: 9-character string representing board state
        
        Returns:
            Position (0-8) to place symbol, or None if no moves available
        """
        available_moves = [i for i, cell in enumerate(board) if cell == '.']
        
        if not available_moves:
            return None
        
        return random.choice(available_moves)


class HeuristicOpponent:
    """
    Opponent using deterministic heuristic strategy
    
    Priority order:
    1. Win immediately if possible
    2. Block opponent's winning move
    3. Create fork (2+ winning threats)
    4. Block opponent's fork
    5. Take center (position 4)
    6. Take opposite corner if opponent has corner
    7. Take any available corner (0,2,6,8)
    8. Take any side position (1,3,5,7)
    """
    
    def __init__(self, symbol: str = 'O'):
        """
        Initialize heuristic opponent
        
        Args:
            symbol: 'X' or 'O' - the symbol this opponent plays
        """
        self.symbol = symbol
        self.opponent_symbol = 'X' if symbol == 'O' else 'O'
        
        # Winning lines (rows, columns, diagonals)
        self.lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        # Board positions
        self.center = 4
        self.corners = [0, 2, 6, 8]
        self.sides = [1, 3, 5, 7]
        self.opposite_corners = {0: 8, 2: 6, 6: 2, 8: 0}
    
    def get_move(self, board: str) -> Optional[int]:
        """
        Select move based on heuristic priority
        
        Args:
            board: 9-character string representing board state
        
        Returns:
            Position (0-8) to place symbol, or None if no moves available
        """
        # 1. Win immediately
        move = self._find_winning_move(board, self.symbol)
        if move is not None:
            return move
        
        # 2. Block opponent's winning move
        move = self._find_winning_move(board, self.opponent_symbol)
        if move is not None:
            return move
        
        # 3. Create a fork
        move = self._find_fork_move(board, self.symbol)
        if move is not None:
            return move
        
        # 4. Block opponent's fork
        move = self._block_fork(board)
        if move is not None:
            return move
        
        # 5. Take center
        if board[self.center] == '.':
            return self.center
        
        # 6. Take opposite corner if opponent took a corner
        move = self._take_opposite_corner(board)
        if move is not None:
            return move
        
        # 7. Take any free corner
        move = self._take_any_corner(board)
        if move is not None:
            return move
        
        # 8. Take any free side
        move = self._take_any_side(board)
        if move is not None:
            return move
        
        return None
    
    def _find_winning_move(self, board: str, symbol: str) -> Optional[int]:
        """
        Find a move that wins the game for given symbol
        
        Args:
            board: 9-character string representing board state
            symbol: 'X' or 'O' to find winning move for
        
        Returns:
            Position of winning move, or None if no winning move exists
        """
        for line in self.lines:
            symbols = [board[i] for i in line]
            if symbols.count(symbol) == 2 and symbols.count('.') == 1:
                # Found winning line with one empty spot
                empty_idx = line[symbols.index('.')]
                return empty_idx
        return None
    
    def _find_fork_move(self, board: str, symbol: str) -> Optional[int]:
        """
        Find a move that creates a fork (two winning opportunities)
        
        A fork occurs when placing a symbol creates two or more lines
        that each have 2 of that symbol and 1 empty position.
        
        Args:
            board: 9-character string representing board state
            symbol: 'X' or 'O' to find fork move for
        
        Returns:
            Position that creates fork, or None if no fork move exists
        """
        available = [i for i, cell in enumerate(board) if cell == '.']
        
        for move in available:
            # Simulate placing symbol
            test_board = list(board)
            test_board[move] = symbol
            test_board = ''.join(test_board)
            
            # Count how many winning opportunities this creates
            winning_lines = 0
            for line in self.lines:
                symbols = [test_board[i] for i in line]
                if symbols.count(symbol) == 2 and symbols.count('.') == 1:
                    winning_lines += 1
            
            # Fork = 2 or more winning opportunities
            if winning_lines >= 2:
                return move
        
        return None
    
    def _block_fork(self, board: str) -> Optional[int]:
        """
        Block opponent's fork attempt
        
        Strategy: If opponent can create fork, either:
        1. Create a two-in-a-row to force opponent to block (if safe)
        2. Block the fork position directly
        
        Args:
            board: 9-character string representing board state
        
        Returns:
            Position to block fork, or None if no fork threat exists
        """
        # First check if opponent can create fork
        opponent_fork = self._find_fork_move(board, self.opponent_symbol)
        
        if opponent_fork is None:
            return None
        
        # Try to create a two-in-a-row to force opponent to block
        available = [i for i, cell in enumerate(board) if cell == '.']
        
        for move in available:
            test_board = list(board)
            test_board[move] = self.symbol
            test_board = ''.join(test_board)
            
            # Check if this creates a threat
            for line in self.lines:
                symbols = [test_board[i] for i in line]
                if symbols.count(self.symbol) == 2 and symbols.count('.') == 1:
                    empty_idx = line[symbols.index('.')]
                    # Make sure blocking doesn't create opponent's fork
                    blocking_board = list(test_board)
                    blocking_board[empty_idx] = self.opponent_symbol
                    blocking_board = ''.join(blocking_board)
                    
                    if self._find_fork_move(blocking_board, self.opponent_symbol) is None:
                        return move
        
        # If can't force block safely, just block the fork position
        return opponent_fork
    
    def _take_opposite_corner(self, board: str) -> Optional[int]:
        """
        Take opposite corner if opponent occupies a corner
        
        Args:
            board: 9-character string representing board state
        
        Returns:
            Opposite corner position, or None if not applicable
        """
        for corner, opposite in self.opposite_corners.items():
            if board[corner] == self.opponent_symbol and board[opposite] == '.':
                return opposite
        return None
    
    def _take_any_corner(self, board: str) -> Optional[int]:
        """
        Take any available corner
        
        Args:
            board: 9-character string representing board state
        
        Returns:
            First available corner position, or None if all corners taken
        """
        available_corners = [c for c in self.corners if board[c] == '.']
        return available_corners[0] if available_corners else None
    
    def _take_any_side(self, board: str) -> Optional[int]:
        """
        Take any available side position
        
        Args:
            board: 9-character string representing board state
        
        Returns:
            First available side position, or None if all sides taken
        """
        available_sides = [s for s in self.sides if board[s] == '.']
        return available_sides[0] if available_sides else None


class SimpleHeuristicOpponent:
    """
    Simplified heuristic opponent - easier to implement and debug
    
    Priority:
    1. Win immediately
    2. Block opponent's win
    3. Take center
    4. Take first available move
    """
    
    def __init__(self, symbol: str = 'O'):
        """
        Initialize simple heuristic opponent
        
        Args:
            symbol: 'X' or 'O' - the symbol this opponent plays
        """
        self.symbol = symbol
        self.opponent_symbol = 'X' if symbol == 'O' else 'O'
        
        self.lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
    
    def get_move(self, board: str) -> Optional[int]:
        """
        Select move using simplified priority
        
        Args:
            board: 9-character string representing board state
        
        Returns:
            Position (0-8) to place symbol, or None if no moves available
        """
        # Win
        move = self._find_winning_move(board, self.symbol)
        if move is not None:
            return move
        
        # Block
        move = self._find_winning_move(board, self.opponent_symbol)
        if move is not None:
            return move
        
        # Center
        if board[4] == '.':
            return 4
        
        # First available move
        available = [i for i, cell in enumerate(board) if cell == '.']
        return available[0] if available else None
    
    def _find_winning_move(self, board: str, symbol: str) -> Optional[int]:
        """Find move that completes a line for symbol"""
        for line in self.lines:
            symbols = [board[i] for i in line]
            if symbols.count(symbol) == 2 and symbols.count('.') == 1:
                return line[symbols.index('.')]
        return None


# Test the implementations
if __name__ == "__main__":
    print("Testing Opponent Policies")
    print("=" * 50)
    
    # Test board state
    board = "X.....O.."
    print(f"\nTest board:\n{board[:3]}\n{board[3:6]}\n{board[6:]}\n")
    
    # Test Random Opponent
    print("Random Opponent:")
    random_opp = RandomOpponent('O')
    for i in range(5):
        move = random_opp.get_move(board)
        print(f"  Move {i+1}: {move}")
    
    # Test Heuristic Opponent
    print("\nHeuristic Opponent:")
    heuristic_opp = HeuristicOpponent('O')
    move = heuristic_opp.get_move(board)
    print(f"  Chosen move: {move}")
    
    # Test winning detection
    board_win = "XX...O.O."
    print(f"\nBoard with winning opportunity:\n{board_win[:3]}\n{board_win[3:6]}\n{board_win[6:]}\n")
    move = heuristic_opp.get_move(board_win)
    print(f"  Heuristic move (should win): {move}")
    
    # Test blocking
    heuristic_opp_x = HeuristicOpponent('X')
    move = heuristic_opp_x.get_move(board_win)
    print(f"  Heuristic move (should block): {move}")
    
    # Test Simple Heuristic
    print("\nSimple Heuristic Opponent:")
    simple_opp = SimpleHeuristicOpponent('O')
    move = simple_opp.get_move(board)
    print(f"  Chosen move: {move}")
