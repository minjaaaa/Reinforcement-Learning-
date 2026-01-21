"""
Evaluation System for Tic-Tac-Toe Agent
Simulates games and generates performance statistics
"""

from typing import Dict, Optional
from game_state import TicTacToeGame


class TicTacToeEvaluator:
    """
    Simulate and evaluate agent performance against various opponents
    """
    
    def __init__(self, agent):
        """
        Initialize evaluator with trained agent
        
        Args:
            agent: Trained ValueIterationAgent
        """
        self.agent = agent
        self.game = TicTacToeGame()
    
    def play_single_game(self, 
                        opponent, 
                        agent_plays_first: bool,
                        verbose: bool = False) -> Optional[str]:
        """
        Simulate one complete game
        
        Args:
            opponent: Opponent policy object
            agent_plays_first: True if agent is X (first), False if O (second)
            verbose: Print game moves if True
        
        Returns:
            'X' for X win, 'O' for O win, 'D' for draw
        """
        board = '.' * 9
        agent_symbol = 'X' if agent_plays_first else 'O'
        opponent_symbol = 'O' if agent_plays_first else 'X'
        
        # Set opponent symbol
        opponent.symbol = opponent_symbol
        if hasattr(opponent, 'opponent_symbol'):
            opponent.opponent_symbol = agent_symbol
        
        current_player = 'X'  # X always goes first
        
        if verbose:
            print("\nStarting game:")
            print(f"Agent: {agent_symbol}, Opponent: {opponent_symbol}")
            self.game.print_board(board)
        
        move_count = 0
        
        while not self.game.is_terminal(board):
            move_count += 1
            
            if current_player == agent_symbol:
                # Agent's turn
                move = self.agent.get_move(board, agent_symbol)
                if move is None:
                    break
                if verbose:
                    print(f"Move {move_count}: Agent plays at position {move}")
            else:
                # Opponent's turn
                move = opponent.get_move(board)
                if move is None:
                    break
                if verbose:
                    print(f"Move {move_count}: Opponent plays at position {move}")
            
            board = self.game.apply_move(board, move, current_player)
            
            if verbose:
                self.game.print_board(board)
            
            current_player = 'O' if current_player == 'X' else 'X'
        
        winner = self.game.check_winner(board)
        
        if verbose:
            if winner == 'D':
                print("Game ended in a draw")
            else:
                print(f"Winner: {winner}")
        
        return winner
    
    def simulate_games(self, 
                      opponent, 
                      n_games: int,
                      agent_plays_first: bool) -> Dict[str, int]:
        """
        Run multiple game simulations
        
        Args:
            opponent: Opponent policy object
            n_games: Number of games to simulate
            agent_plays_first: True if agent plays as X, False if O
        
        Returns:
            Dictionary with win/loss/draw counts
        """
        results = {
            'agent_wins': 0,
            'opponent_wins': 0,
            'draws': 0,
            'total_games': n_games
        }
        
        agent_symbol = 'X' if agent_plays_first else 'O'
        opponent_symbol = 'O' if agent_plays_first else 'X'
        
        for game_num in range(n_games):
            winner = self.play_single_game(opponent, agent_plays_first, verbose=False)
            
            if winner == agent_symbol:
                results['agent_wins'] += 1
            elif winner == opponent_symbol:
                results['opponent_wins'] += 1
            else:  # Draw
                results['draws'] += 1
            
            # Print progress every 100 games
            if (game_num + 1) % 100 == 0:
                print(f"  Progress: {game_num + 1}/{n_games} games completed", end='\r')
        
        print(f"  Progress: {n_games}/{n_games} games completed")
        
        return results
    
    def evaluate_agent(self, 
                      opponents: Dict[str, object],
                      n_games: int = 1000) -> Dict:
        """
        Complete evaluation protocol against multiple opponents
        
        Args:
            opponents: Dictionary of {'opponent_name': opponent_object}
            n_games: Number of games per scenario
        
        Returns:
            Nested dictionary with all results
        """
        evaluation_results = {}
        
        for opp_name, opponent in opponents.items():
            print(f"\n{'='*60}")
            print(f"Evaluating against {opp_name}")
            print(f"{'='*60}")
            
            # Agent plays first (X)
            print(f"\nAgent plays FIRST (X) vs {opp_name} (O):")
            results_first = self.simulate_games(opponent, n_games, agent_plays_first=True)
            self._print_results(results_first, n_games)
            
            # Agent plays second (O)
            print(f"\nAgent plays SECOND (O) vs {opp_name} (X):")
            results_second = self.simulate_games(opponent, n_games, agent_plays_first=False)
            self._print_results(results_second, n_games)
            
            # Store results
            evaluation_results[opp_name] = {
                'agent_first': results_first,
                'agent_second': results_second
            }
        
        return evaluation_results
    
    def _print_results(self, results: Dict[str, int], n_games: int):
        """
        Print formatted statistics
        
        Args:
            results: Dictionary with win/loss/draw counts
            n_games: Total number of games
        """
        agent_wins = results['agent_wins']
        opponent_wins = results['opponent_wins']
        draws = results['draws']
        
        print(f"  Agent Wins:    {agent_wins:4d} ({agent_wins/n_games*100:5.1f}%)")
        print(f"  Opponent Wins: {opponent_wins:4d} ({opponent_wins/n_games*100:5.1f}%)")
        print(f"  Draws:         {draws:4d} ({draws/n_games*100:5.1f}%)")
        print(f"  Total Games:   {n_games}")
    
    def generate_report(self, 
                       evaluation_results: Dict,
                       output_file: str = "evaluation_report.txt"):
        """
        Generate comprehensive text report
        
        Args:
            evaluation_results: Results from evaluate_agent
            output_file: Path to save report
        """
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TIC-TAC-TOE AGENT EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Agent Symbol: {self.agent.agent_symbol}\n")
            f.write(f"Gamma: {self.agent.gamma}\n")
            f.write(f"Theta: {self.agent.theta}\n")
            f.write(f"Opponent Type (Training): {self.agent.opponent.__class__.__name__}\n\n")
            
            for opp_name, results in evaluation_results.items():
                f.write(f"\n{'='*70}\n")
                f.write(f"Opponent: {opp_name}\n")
                f.write(f"{'='*70}\n")
                
                # Agent plays first
                f.write("\nAgent plays FIRST (X):\n")
                f.write("-"*70 + "\n")
                r1 = results['agent_first']
                total = r1['total_games']
                f.write(f"  Wins:   {r1['agent_wins']:4d} ({r1['agent_wins']/total*100:5.1f}%)\n")
                f.write(f"  Losses: {r1['opponent_wins']:4d} ({r1['opponent_wins']/total*100:5.1f}%)\n")
                f.write(f"  Draws:  {r1['draws']:4d} ({r1['draws']/total*100:5.1f}%)\n")
                
                # Agent plays second
                f.write("\nAgent plays SECOND (O):\n")
                f.write("-"*70 + "\n")
                r2 = results['agent_second']
                total = r2['total_games']
                f.write(f"  Wins:   {r2['agent_wins']:4d} ({r2['agent_wins']/total*100:5.1f}%)\n")
                f.write(f"  Losses: {r2['opponent_wins']:4d} ({r2['opponent_wins']/total*100:5.1f}%)\n")
                f.write(f"  Draws:  {r2['draws']:4d} ({r2['draws']/total*100:5.1f}%)\n")
                
                # Combined statistics
                f.write("\nCombined Statistics:\n")
                f.write("-"*70 + "\n")
                total_wins = r1['agent_wins'] + r2['agent_wins']
                total_losses = r1['opponent_wins'] + r2['opponent_wins']
                total_draws = r1['draws'] + r2['draws']
                total_games = total_wins + total_losses + total_draws
                
                f.write(f"  Total Wins:   {total_wins:4d} ({total_wins/total_games*100:5.1f}%)\n")
                f.write(f"  Total Losses: {total_losses:4d} ({total_losses/total_games*100:5.1f}%)\n")
                f.write(f"  Total Draws:  {total_draws:4d} ({total_draws/total_games*100:5.1f}%)\n")
                f.write(f"  Total Games:  {total_games}\n")
                
                f.write("\n")
        
        print(f"\nReport saved to: {output_file}")
    
    def print_summary(self, evaluation_results: Dict):
        """
        Print summary of all evaluations
        
        Args:
            evaluation_results: Results from evaluate_agent
        """
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        for opp_name, opp_results in evaluation_results.items():
            print(f"\n{opp_name}:")
            print("-"*70)
            
            # Combined statistics
            r1 = opp_results['agent_first']
            r2 = opp_results['agent_second']
            
            total_wins = r1['agent_wins'] + r2['agent_wins']
            total_losses = r1['opponent_wins'] + r2['opponent_wins']
            total_draws = r1['draws'] + r2['draws']
            total_games = total_wins + total_losses + total_draws
            
            print(f"  Overall Record: {total_wins}W - {total_losses}L - {total_draws}D")
            print(f"  Win Rate:  {total_wins/total_games*100:5.1f}%")
            print(f"  Loss Rate: {total_losses/total_games*100:5.1f}%")
            print(f"  Draw Rate: {total_draws/total_games*100:5.1f}%")
            
            print(f"\n  As First Player (X):  {r1['agent_wins']}W - {r1['opponent_wins']}L - {r1['draws']}D")
            print(f"  As Second Player (O): {r2['agent_wins']}W - {r2['opponent_wins']}L - {r2['draws']}D")


# Quick evaluation function
def quick_evaluation(agent, n_games: int = 100):
    """
    Quick evaluation function for testing
    
    Args:
        agent: Trained ValueIterationAgent
        n_games: Number of games to test
    """
    from opponent_policies import RandomOpponent, HeuristicOpponent
    
    evaluator = TicTacToeEvaluator(agent)
    
    print("\n" + "="*60)
    print("QUICK EVALUATION")
    print("="*60)
    
    # Test against random
    print("\n=== VS RANDOM OPPONENT ===")
    random_opp = RandomOpponent()
    
    print("\nAgent as X:")
    r1 = evaluator.simulate_games(random_opp, n_games, agent_plays_first=True)
    print(f"  W: {r1['agent_wins']}, L: {r1['opponent_wins']}, D: {r1['draws']}")
    
    print("\nAgent as O:")
    r2 = evaluator.simulate_games(random_opp, n_games, agent_plays_first=False)
    print(f"  W: {r2['agent_wins']}, L: {r2['opponent_wins']}, D: {r2['draws']}")
    
    # Test against heuristic
    print("\n=== VS HEURISTIC OPPONENT ===")
    heuristic_opp = HeuristicOpponent()
    
    print("\nAgent as X:")
    r3 = evaluator.simulate_games(heuristic_opp, n_games, agent_plays_first=True)
    print(f"  W: {r3['agent_wins']}, L: {r3['opponent_wins']}, D: {r3['draws']}")
    
    print("\nAgent as O:")
    r4 = evaluator.simulate_games(heuristic_opp, n_games, agent_plays_first=False)
    print(f"  W: {r4['agent_wins']}, L: {r4['opponent_wins']}, D: {r4['draws']}")


# Test the implementation
if __name__ == "__main__":
    from opponent_policies import RandomOpponent, HeuristicOpponent
    from value_iteration_agent import ValueIterationAgent
    
    print("Testing Evaluation System")
    print("="*60)
    
    # Create and train a quick agent
    print("\nTraining test agent...")
    random_opp = RandomOpponent('O')
    agent = ValueIterationAgent(random_opp, agent_symbol='X', gamma=0.9, theta=0.01)
    agent.train(max_iterations=50)
    
    # Quick evaluation
    quick_evaluation(agent, n_games=100)
