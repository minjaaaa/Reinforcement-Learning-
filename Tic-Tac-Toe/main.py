"""
Main Training and Evaluation Orchestration
Train agents against different opponents and evaluate performance
"""

import argparse
import os
from datetime import datetime
from typing import Optional

from game_state import TicTacToeGame
from opponent_policies import RandomOpponent, HeuristicOpponent, SimpleHeuristicOpponent
from value_iteration_agent import ValueIterationAgent
from evaluation import TicTacToeEvaluator
from utils import save_training_stats, create_training_stats


def train_agent_vs_random(role: str = 'X', 
                          gamma: float = 0.9, 
                          theta: float = 0.001,
                          max_iterations: int = 1000,
                          save_path: Optional[str] = None):
    """
    Train agent against random opponent
    
    Args:
        role: 'X' (first player) or 'O' (second player)
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum training iterations
        save_path: Path to save trained policy
    
    Returns:
        Trained ValueIterationAgent
    """
    print("\n" + "="*70)
    print("TRAINING AGENT VS RANDOM OPPONENT")
    print("="*70)
    
    # Create opponent (plays opposite role)
    opponent_symbol = 'O' if role == 'X' else 'X'
    opponent = RandomOpponent(opponent_symbol)
    
    # Create agent
    agent = ValueIterationAgent(
        opponent=opponent,
        agent_symbol=role,
        gamma=gamma,
        theta=theta
    )
    
    # Train
    converged, iterations = agent.train(max_iterations=max_iterations)
    
    # Save policy
    if save_path is None:
        save_path = f"agent_vs_random_{role}.pkl"
    
    agent.save_policy(save_path)
    
    # Save training stats
    stats = create_training_stats(
        agent_symbol=role,
        opponent_type='RandomOpponent',
        iterations=iterations,
        converged=converged,
        gamma=gamma,
        theta=theta
    )
    save_training_stats(stats, save_path.replace('.pkl', '_stats.json'))
    
    return agent


def train_agent_vs_heuristic(role: str = 'X',
                             gamma: float = 0.9,
                             theta: float = 0.001,
                             max_iterations: int = 1000,
                             save_path: Optional[str] = None):
    """
    Train agent against heuristic opponent
    
    Args:
        role: 'X' (first player) or 'O' (second player)
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum training iterations
        save_path: Path to save trained policy
    
    Returns:
        Trained ValueIterationAgent
    """
    print("\n" + "="*70)
    print("TRAINING AGENT VS HEURISTIC OPPONENT")
    print("="*70)
    
    # Create opponent (plays opposite role)
    opponent_symbol = 'O' if role == 'X' else 'X'
    opponent = HeuristicOpponent(opponent_symbol)
    
    # Create agent
    agent = ValueIterationAgent(
        opponent=opponent,
        agent_symbol=role,
        gamma=gamma,
        theta=theta
    )
    
    # Train
    converged, iterations = agent.train(max_iterations=max_iterations)
    
    # Save policy
    if save_path is None:
        save_path = f"agent_vs_heuristic_{role}.pkl"
    
    agent.save_policy(save_path)
    
    # Save training stats
    stats = create_training_stats(
        agent_symbol=role,
        opponent_type='HeuristicOpponent',
        iterations=iterations,
        converged=converged,
        gamma=gamma,
        theta=theta
    )
    save_training_stats(stats, save_path.replace('.pkl', '_stats.json'))
    
    return agent


def evaluate_all(agents_to_test: Optional[list] = None, n_games: int = 1000):
    """
    Complete evaluation workflow
    
    Args:
        agents_to_test: List of (agent, description) tuples. If None, load default agents
        n_games: Number of games per scenario
    """
    print("\n" + "="*70)
    print("EVALUATING TRAINED AGENTS")
    print("="*70)
    
    # If no agents provided, try to load saved agents
    if agents_to_test is None:
        agents_to_test = []
        
        # Try to load agents
        agent_files = [
            ('agent_vs_random_X.pkl', 'Agent trained vs Random as X'),
            ('agent_vs_random_O.pkl', 'Agent trained vs Random as O'),
            ('agent_vs_heuristic_X.pkl', 'Agent trained vs Heuristic as X'),
            ('agent_vs_heuristic_O.pkl', 'Agent trained vs Heuristic as O'),
        ]
        
        for filename, description in agent_files:
            if os.path.exists(filename):
                print(f"\nLoading {filename}...")
                # Need to create dummy opponent for loading
                dummy_opp = RandomOpponent()
                agent = ValueIterationAgent(dummy_opp, 'X')
                agent.load_policy(filename)
                agents_to_test.append((agent, description))
    
    if not agents_to_test:
        print("No agents to evaluate. Train agents first.")
        return
    
    # Create opponents for evaluation
    opponents = {
        'Random Policy': RandomOpponent(),
        'Heuristic Policy': HeuristicOpponent(),
        'Simple Heuristic': SimpleHeuristicOpponent()
    }
    
    # Evaluate each agent
    all_results = {}
    
    for agent, description in agents_to_test:
        print(f"\n{'='*70}")
        print(f"Agent: {description}")
        print(f"{'='*70}")
        
        evaluator = TicTacToeEvaluator(agent)
        results = evaluator.evaluate_agent(opponents, n_games=n_games)
        
        # Generate report
        report_name = description.replace(' ', '_').replace('(', '').replace(')', '') + '_report.txt'
        evaluator.generate_report(results, report_name)
        
        # Print summary
        evaluator.print_summary(results)
        
        all_results[description] = results
    
    return all_results


def full_pipeline(gamma: float = 0.9, 
                  theta: float = 0.001,
                  max_iterations: int = 1000,
                  n_eval_games: int = 1000):
    """
    Complete training and evaluation pipeline
    
    Trains agents in all scenarios and evaluates them
    
    Args:
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum training iterations
        n_eval_games: Number of evaluation games
    """
    print("\n" + "="*70)
    print("FULL TRAINING AND EVALUATION PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    agents_to_test = []
    
    # Train agent vs random as X
    print("\n[1/4] Training agent vs Random opponent as X...")
    agent1 = train_agent_vs_random('X', gamma, theta, max_iterations)
    agents_to_test.append((agent1, 'Trained vs Random as X'))
    
    # Train agent vs random as O
    print("\n[2/4] Training agent vs Random opponent as O...")
    agent2 = train_agent_vs_random('O', gamma, theta, max_iterations)
    agents_to_test.append((agent2, 'Trained vs Random as O'))
    
    # Train agent vs heuristic as X
    print("\n[3/4] Training agent vs Heuristic opponent as X...")
    agent3 = train_agent_vs_heuristic('X', gamma, theta, max_iterations)
    agents_to_test.append((agent3, 'Trained vs Heuristic as X'))
    
    # Train agent vs heuristic as O
    print("\n[4/4] Training agent vs Heuristic opponent as O...")
    agent4 = train_agent_vs_heuristic('O', gamma, theta, max_iterations)
    agents_to_test.append((agent4, 'Trained vs Heuristic as O'))
    
    # Evaluate all agents
    print("\n" + "="*70)
    print("TRAINING COMPLETE - STARTING EVALUATION")
    print("="*70)
    
    results = evaluate_all(agents_to_test, n_eval_games)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return results


def interactive_play(agent_file: Optional[str] = None):
    """
    Play interactively against a trained agent
    
    Args:
        agent_file: Path to trained agent policy file
    """
    from game_state import TicTacToeGame
    
    print("\n" + "="*70)
    print("INTERACTIVE PLAY MODE")
    print("="*70)
    
    # Load agent
    if agent_file is None or not os.path.exists(agent_file):
        print("No agent file provided or file not found.")
        print("Training a quick agent...")
        opponent = RandomOpponent('O')
        agent = ValueIterationAgent(opponent, 'X', gamma=0.9, theta=0.01)
        agent.train(max_iterations=100)
    else:
        print(f"Loading agent from {agent_file}...")
        dummy_opp = RandomOpponent()
        agent = ValueIterationAgent(dummy_opp, 'X')
        agent.load_policy(agent_file)
    
    game = TicTacToeGame()
    
    # Choose who goes first
    human_first = input("\nDo you want to go first? (y/n): ").lower().startswith('y')
    
    if human_first:
        human_symbol = 'X'
        agent_symbol = 'O'
        print(f"\nYou are {human_symbol}, Agent is {agent_symbol}")
    else:
        human_symbol = 'O'
        agent_symbol = 'X'
        print(f"\nYou are {human_symbol}, Agent is {agent_symbol}")
    
    board = '.' * 9
    current_player = 'X'  # X always starts
    
    print("\nBoard positions:")
    game.print_board("012345678")
    
    while not game.is_terminal(board):
        print(f"\nCurrent board:")
        game.print_board(board)
        
        if current_player == human_symbol:
            # Human's turn
            while True:
                try:
                    move = int(input(f"Your move (0-8): "))
                    if move in game.get_available_moves(board):
                        break
                    else:
                        print("Invalid move. Position already taken or out of range.")
                except ValueError:
                    print("Please enter a number between 0 and 8.")
        else:
            # Agent's turn
            move = agent.get_move(board, agent_symbol)
            print(f"Agent plays at position {move}")
        
        board = game.apply_move(board, move, current_player)
        current_player = 'O' if current_player == 'X' else 'X'
    
    # Game over
    print("\nFinal board:")
    game.print_board(board)
    
    winner = game.check_winner(board)
    if winner == 'D':
        print("It's a draw!")
    elif winner == human_symbol:
        print("You win! Congratulations!")
    else:
        print("Agent wins!")


def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Tic-Tac-Toe Value Iteration Training and Evaluation')
    
    parser.add_argument('--train', type=str, choices=['random', 'heuristic', 'both'],
                       help='Train agent against specified opponent')
    parser.add_argument('--role', type=str, choices=['X', 'O', 'both'], default='X',
                       help='Agent role: X (first player) or O (second player)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained agents')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run complete training and evaluation pipeline')
    parser.add_argument('--play', action='store_true',
                       help='Play interactively against trained agent')
    parser.add_argument('--agent-file', type=str,
                       help='Path to agent policy file (for --play mode)')
    parser.add_argument('--gamma', type=float, default=0.9,
                       help='Discount factor (default: 0.9)')
    parser.add_argument('--theta', type=float, default=0.001,
                       help='Convergence threshold (default: 0.001)')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Maximum training iterations (default: 1000)')
    parser.add_argument('--games', type=int, default=1000,
                       help='Number of evaluation games (default: 1000)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute requested action
    if args.full_pipeline:
        full_pipeline(args.gamma, args.theta, args.iterations, args.games)
    
    elif args.train:
        roles = ['X', 'O'] if args.role == 'both' else [args.role]
        
        for role in roles:
            if args.train == 'random':
                train_agent_vs_random(role, args.gamma, args.theta, args.iterations)
            elif args.train == 'heuristic':
                train_agent_vs_heuristic(role, args.gamma, args.theta, args.iterations)
            elif args.train == 'both':
                train_agent_vs_random(role, args.gamma, args.theta, args.iterations)
                train_agent_vs_heuristic(role, args.gamma, args.theta, args.iterations)
    
    elif args.evaluate:
        evaluate_all(n_games=args.games)
    
    elif args.play:
        interactive_play(args.agent_file)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    # If run without arguments, execute full pipeline with reduced iterations for demo
    import sys
    
    if len(sys.argv) == 1:
        print("No arguments provided. Running demo with reduced iterations...")
        print("For full usage, run: python main.py --help")
        print()
        
        # Quick demo
        print("="*70)
        print("DEMO MODE - Quick Training and Evaluation")
        print("="*70)
        
        # Train one agent
        agent = train_agent_vs_random('X', gamma=0.9, theta=0.01, max_iterations=100)
        
        # Quick evaluation
        from evaluation import quick_evaluation
        quick_evaluation(agent, n_games=100)
        
        print("\nFor full training, run:")
        print("  python main.py --full-pipeline")
    else:
        main()
