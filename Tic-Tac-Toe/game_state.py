"""
Tic-Tac-Toe Game State Implementation
Handles board representation, game rules, and state generation
"""

from typing import List, Set, Optional
from collections import deque


class TicTacToeGame:
    """
    Tic-Tac-Toe game state manager
    
    Board representation: 9-character string (e.g., "XO.X.....")
    Positions 0-8, left-to-right, top-to-bottom:
      0 | 1 | 2
      ---------
      3 | 4 | 5
      ---------
      6 | 7 | 8
    """
    
    def __init__(self):
        """Initialize game with winning lines"""
        # Winning combinations
        self.lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
    def is_terminal(self, board: str) -> bool:
        """
        Check if game is over (win or draw)
        
        Args:
            board: 9-character string representing board state
            
        Returns:
            True if game is over, False otherwise
        """
        return self.check_winner(board) is not None
    
    def check_winner(self, board: str) -> Optional[str]:
        """
        Check game outcome
        
        Args:
            board: 9-character string representing board state
            
        Returns:
            'X' if X wins, 'O' if O wins, 'D' for draw, None if ongoing
        """
        # Check all winning lines
        for line in self.lines:
            symbols = [board[i] for i in line]
            if symbols[0] != '.' and symbols[0] == symbols[1] == symbols[2]:
                return symbols[0]
        
        # Check for draw (board full with no winner)
        if '.' not in board:
            return 'D'
        
        # Game is still ongoing
        return None
    
    def get_available_moves(self, board: str) -> List[int]:
        """
        Get list of empty positions
        
        Args:
            board: 9-character string representing board state
            
        Returns:
            List of available positions [0-8]
        """
        return [i for i, cell in enumerate(board) if cell == '.']
    
    def apply_move(self, board: str, position: int, symbol: str) -> str:
        """
        Apply a move to the board
        
        Args:
            board: 9-character string representing board state
            position: Position to place symbol (0-8)
            symbol: 'X' or 'O'
            
        Returns:
            New board state as string
        """
        board_list = list(board)
        board_list[position] = symbol
        return ''.join(board_list)
    
    def get_reward(self, board: str, agent_symbol: str) -> float:
        """
        Get reward for agent in given state
        
        Args:
            board: 9-character string representing board state
            agent_symbol: 'X' or 'O' - the agent's symbol
            
        Returns:
            +1.0 if agent wins, -1.0 if agent loses, 0.0 otherwise
        """
        winner = self.check_winner(board)
        
        if winner == agent_symbol:
            return 1.0
        elif winner is not None and winner != 'D':
            return -1.0
        else:
            return 0.0
    
    def get_all_states(self) -> Set[str]:
        """
        Generate all possible valid board states using BFS
        
        This explores all reachable states from empty board,
        considering both X and O turns
        
        Returns:
            Set of all valid board state strings
        """
        all_states = set()
        queue = deque(['.' * 9])
        all_states.add('.' * 9)
        
        while queue:
            current_board = queue.popleft()
            
            # Skip if terminal state
            if self.is_terminal(current_board):
                continue
            
            # Determine whose turn it is (X starts, then alternates)
            x_count = current_board.count('X')
            o_count = current_board.count('O')
            
            # X goes first, so if counts are equal, it's X's turn
            current_symbol = 'X' if x_count == o_count else 'O'
            
            # Try all available moves
            for move in self.get_available_moves(current_board):
                new_board = self.apply_move(current_board, move, current_symbol)
                
                if new_board not in all_states:
                    all_states.add(new_board)
                    queue.append(new_board)
        
        return all_states
    
    def print_board(self, board: str):
        """
        Pretty print board in 3x3 format
        
        Args:
            board: 9-character string representing board state
        """
        print()
        for i in range(3):
            row = board[i*3:(i+1)*3]
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("-----------")
        print()
    
    def count_symbols(self, board: str) -> tuple:
        """
        Count X and O symbols on board
        
        Args:
            board: 9-character string representing board state
            
        Returns:
            Tuple of (x_count, o_count)
        """
        return (board.count('X'), board.count('O'))
    
    def whose_turn(self, board: str) -> str:
        """
        Determine whose turn it is
        
        Args:
            board: 9-character string representing board state
            
        Returns:
            'X' or 'O' indicating current player
        """
        x_count, o_count = self.count_symbols(board)
        return 'X' if x_count == o_count else 'O'


# Test the implementation
if __name__ == "__main__":
    game = TicTacToeGame()
    
    # Test 1: Empty board
    board = '.' * 9
    print("Test 1: Empty board")
    game.print_board(board)
    print(f"Terminal: {game.is_terminal(board)}")
    print(f"Winner: {game.check_winner(board)}")
    print(f"Available moves: {game.get_available_moves(board)}")
    
    # Test 2: X wins horizontally
    board = "XXX...OO."
    print("\nTest 2: X wins")
    game.print_board(board)
    print(f"Terminal: {game.is_terminal(board)}")
    print(f"Winner: {game.check_winner(board)}")
    
    # Test 3: Draw
    board = "XXOOOXXXO"
    print("\nTest 3: Draw")
    game.print_board(board)
    print(f"Terminal: {game.is_terminal(board)}")
    print(f"Winner: {game.check_winner(board)}")
    
    # Test 4: Generate all states
    print("\nTest 4: Generating all states...")
    all_states = game.get_all_states()
    print(f"Total valid states: {len(all_states)}")
