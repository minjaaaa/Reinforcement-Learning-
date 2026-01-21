"""
Utility Functions for Tic-Tac-Toe Project
Helper functions for board manipulation, visualization, and data persistence
"""

import json
import pickle
from typing import List, Dict, Any


def board_to_string(board_list: List[str]) -> str:
    """
    Convert list representation to string representation
    
    Args:
        board_list: List of 9 strings ('X', 'O', or '.')
    
    Returns:
        9-character string representation
    """
    return ''.join(board_list)


def string_to_board(board_str: str) -> List[List[str]]:
    """
    Convert string to 2D list representation
    
    Args:
        board_str: 9-character string representation
    
    Returns:
        3x3 list representation
    """
    return [
        [board_str[0], board_str[1], board_str[2]],
        [board_str[3], board_str[4], board_str[5]],
        [board_str[6], board_str[7], board_str[8]]
    ]


def visualize_board(board: str, show_indices: bool = False):
    """
    Print board with nice formatting
    
    Args:
        board: 9-character string representing board state
        show_indices: If True, show position indices for empty cells
    """
    print()
    for i in range(3):
        row = []
        for j in range(3):
            idx = i * 3 + j
            cell = board[idx]
            if cell == '.' and show_indices:
                row.append(str(idx))
            else:
                row.append(cell if cell != '.' else ' ')
        print(f" {row[0]} | {row[1]} | {row[2]} ")
        if i < 2:
            print("-----------")
    print()


def get_symmetric_states(board: str) -> List[str]:
    """
    Generate symmetric/rotated board states
    Can be used to reduce state space for faster training
    
    Symmetries:
    - 3 rotations (90°, 180°, 270°)
    - 4 reflections (horizontal, vertical, 2 diagonals)
    
    Args:
        board: 9-character string representing board state
    
    Returns:
        List of symmetric board states
    """
    def rotate_90(b: str) -> str:
        """Rotate board 90 degrees clockwise"""
        return ''.join([b[6], b[3], b[0],
                       b[7], b[4], b[1],
                       b[8], b[5], b[2]])
    
    def reflect_horizontal(b: str) -> str:
        """Reflect board horizontally"""
        return ''.join([b[2], b[1], b[0],
                       b[5], b[4], b[3],
                       b[8], b[7], b[6]])
    
    def reflect_vertical(b: str) -> str:
        """Reflect board vertically"""
        return ''.join([b[6], b[7], b[8],
                       b[3], b[4], b[5],
                       b[0], b[1], b[2]])
    
    def reflect_diagonal_main(b: str) -> str:
        """Reflect board along main diagonal (top-left to bottom-right)"""
        return ''.join([b[0], b[3], b[6],
                       b[1], b[4], b[7],
                       b[2], b[5], b[8]])
    
    def reflect_diagonal_anti(b: str) -> str:
        """Reflect board along anti-diagonal (top-right to bottom-left)"""
        return ''.join([b[8], b[5], b[2],
                       b[7], b[4], b[1],
                       b[6], b[3], b[0]])
    
    symmetric_states = [board]
    
    # Rotations
    rotated_90 = rotate_90(board)
    rotated_180 = rotate_90(rotated_90)
    rotated_270 = rotate_90(rotated_180)
    
    symmetric_states.extend([rotated_90, rotated_180, rotated_270])
    
    # Reflections
    symmetric_states.append(reflect_horizontal(board))
    symmetric_states.append(reflect_vertical(board))
    symmetric_states.append(reflect_diagonal_main(board))
    symmetric_states.append(reflect_diagonal_anti(board))
    
    # Remove duplicates and return
    return list(set(symmetric_states))


def save_training_stats(stats: Dict[str, Any], filename: str):
    """
    Save training statistics to JSON file
    
    Args:
        stats: Dictionary containing training statistics
        filename: Path to save file
    """
    if not filename.endswith('.json'):
        filename += '.json'
    
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Training stats saved to: {filename}")


def load_training_stats(filename: str) -> Dict[str, Any]:
    """
    Load training statistics from JSON file
    
    Args:
        filename: Path to stats file
    
    Returns:
        Dictionary containing training statistics
    """
    with open(filename, 'r') as f:
        stats = json.load(f)
    
    return stats


def create_training_stats(agent_symbol: str, 
                         opponent_type: str,
                         iterations: int,
                         converged: bool,
                         gamma: float,
                         theta: float) -> Dict[str, Any]:
    """
    Create training statistics dictionary
    
    Args:
        agent_symbol: 'X' or 'O'
        opponent_type: Name of opponent class
        iterations: Number of iterations completed
        converged: Whether training converged
        gamma: Discount factor used
        theta: Convergence threshold used
    
    Returns:
        Dictionary with training statistics
    """
    return {
        'agent_symbol': agent_symbol,
        'opponent_type': opponent_type,
        'iterations': iterations,
        'converged': converged,
        'gamma': gamma,
        'theta': theta
    }


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format float as percentage string
    
    Args:
        value: Float between 0 and 1
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def print_board_comparison(board1: str, board2: str, 
                          title1: str = "Board 1", 
                          title2: str = "Board 2"):
    """
    Print two boards side by side for comparison
    
    Args:
        board1: First board state
        board2: Second board state
        title1: Title for first board
        title2: Title for second board
    """
    print(f"\n{title1:^15} | {title2:^15}")
    print("-" * 33)
    
    for i in range(3):
        row1 = [board1[i*3+j] if board1[i*3+j] != '.' else ' ' for j in range(3)]
        row2 = [board2[i*3+j] if board2[i*3+j] != '.' else ' ' for j in range(3)]
        
        print(f" {row1[0]} | {row1[1]} | {row1[2]}   |   {row2[0]} | {row2[1]} | {row2[2]} ")
        
        if i < 2:
            print("-------+-------+-------")
    print()


def count_states_by_type(all_states: set, game) -> Dict[str, int]:
    """
    Categorize and count states
    
    Args:
        all_states: Set of all board states
        game: TicTacToeGame instance
    
    Returns:
        Dictionary with state counts by category
    """
    counts = {
        'total': len(all_states),
        'terminal': 0,
        'x_wins': 0,
        'o_wins': 0,
        'draws': 0,
        'x_turn': 0,
        'o_turn': 0
    }
    
    for state in all_states:
        if game.is_terminal(state):
            counts['terminal'] += 1
            winner = game.check_winner(state)
            if winner == 'X':
                counts['x_wins'] += 1
            elif winner == 'O':
                counts['o_wins'] += 1
            elif winner == 'D':
                counts['draws'] += 1
        else:
            turn = game.whose_turn(state)
            if turn == 'X':
                counts['x_turn'] += 1
            else:
                counts['o_turn'] += 1
    
    return counts


def validate_board(board: str) -> bool:
    """
    Validate that board state is legal
    
    Checks:
    - Length is 9
    - Only contains 'X', 'O', '.'
    - X and O counts differ by at most 1
    - X count >= O count (X goes first)
    - Only one player can win
    
    Args:
        board: 9-character string representing board state
    
    Returns:
        True if valid, False otherwise
    """
    # Check length
    if len(board) != 9:
        return False
    
    # Check characters
    if not all(c in 'XO.' for c in board):
        return False
    
    # Check counts
    x_count = board.count('X')
    o_count = board.count('O')
    
    if x_count < o_count or x_count > o_count + 1:
        return False
    
    # Check for double win (both players can't win)
    from game_state import TicTacToeGame
    game = TicTacToeGame()
    
    x_wins = False
    o_wins = False
    
    for line in game.lines:
        symbols = [board[i] for i in line]
        if symbols[0] == 'X' and symbols[0] == symbols[1] == symbols[2]:
            x_wins = True
        if symbols[0] == 'O' and symbols[0] == symbols[1] == symbols[2]:
            o_wins = True
    
    if x_wins and o_wins:
        return False
    
    return True


# Test the utilities
if __name__ == "__main__":
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Test board conversion
    board_str = "XOX.O...."
    print(f"\nOriginal board string: {board_str}")
    
    board_2d = string_to_board(board_str)
    print("2D representation:")
    for row in board_2d:
        print(row)
    
    # Test visualization
    print("\nVisualization:")
    visualize_board(board_str)
    
    print("With indices:")
    visualize_board(board_str, show_indices=True)
    
    # Test symmetric states
    print("Symmetric states:")
    symmetric = get_symmetric_states(board_str)
    print(f"Found {len(symmetric)} unique symmetric states")
    for i, state in enumerate(symmetric, 1):
        print(f"  {i}. {state}")
    
    # Test validation
    print("\nValidation tests:")
    valid_boards = [".........", "X........", "XO.......", "XOXOXOXOX"]
    invalid_boards = ["XXX", "XXXXXXXXX", "OX.......", "XXXOOOXXX"]
    
    for board in valid_boards:
        print(f"  {board}: {validate_board(board)}")
    
    for board in invalid_boards:
        print(f"  {board}: {validate_board(board)}")
    
    # Test stats
    print("\nCreating training stats...")
    stats = create_training_stats('X', 'RandomOpponent', 50, True, 0.9, 0.001)
    print(json.dumps(stats, indent=2))
