# main.py
import sys
import os
import pygame
import argparse
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from UI.game_ui import BoardGameUI, GameManager, LobbyScreen, GameOverScreen, StatsScreen
from Agent import Agent
from Board import Board


def print_snake_vision(view, step, board_size):
    """Prints the snake view in a simple cross layout with only board characters"""
    # Extract only the last character (0, W, S, G, R)
    clean = [[str(cell).strip()[-1] for cell in arm] for arm in view]
    up, down, left, right = clean
    
    # Build horizontal part to calculate indentation
    left_str = "".join(reversed(left))
    right_str = "".join(right)
    indent = " " * len(left_str)
    
    print(f"\n--- Step {step} ---")
    # Print UP (farthest cell first)
    for cell in reversed(up):
        print(f"{indent}{cell}")
    
    # Print Middle Row
    print(f"{left_str}H{right_str}")
    
    # Print DOWN (closes cell first)
    for cell in down:
        print(f"{indent}{cell}")


def train_agent(num_episodes=100, board_size=10, visualize=False, fps=10, agent=None, start_episode=0, debug=False):
    """
    Train the DQN agent
    
    Args:
        num_episodes: Number of training episodes
        board_size: Size of the board (10, 14, or 18)
        visualize: Whether to show the game UI (simple visualization)
        fps: Frames per second (only if visualize=True)
        agent: Pre-loaded agent (if continuing training)
        start_episode: Starting episode number (for continued training)
        debug: Enable debug prints
    """
    # Initialize agent if not provided
    if agent is None:
        agent = Agent(board_size=board_size, debug=debug)
        initial_epsilon = 1.0
    else:
        # Starting low for pre-trained models
        agent.debug = debug
        initial_epsilon = 0.005
    
    epsilon = initial_epsilon
    
    epsilon_decay = 0.995  # Slower decay
    
    # Statistics
    stats = {
        'total_episodes': 0,
        'total_score': 0,
        'high_score': 0,
        'scores': []
    }
    
    directions_map = ['UP', 'DOWN', 'LEFT', 'RIGHT']  # Match Agent output indices
    
    print(f"\n{'='*60}")
    if start_episode > 0:
        print(f"Continuing training from episode {start_episode + 1}")
    print(f"Training: {num_episodes} episodes on {board_size}x{board_size} board")
    if debug:
        print(f"🔧 DEBUG MODE ENABLED")
    print(f"{'='*60}\n")
    
    for episode_num in range(num_episodes):
        print(f"\n=== Episode {start_episode + episode_num + 1} | Epsilon: {epsilon:.3f} ===")
        
        # Enable debug for first 2 episodes
        if debug and episode_num < 2:
            agent.debug = True
            agent.train_count = 0  # Reset counter for each episode debug
        else:
            agent.debug = False
        
        # Initialize new game for this episode
        if visualize:
            game_ui = BoardGameUI(board_size + 2)  # +2 for walls
            board = game_ui.board
        else:
            # Create board without UI
            board = Board(board_size + 2)
            board.initialize()
        
        step_count = 0
        episode_reward = 0
        max_steps = board_size * board_size * 20  # Límite: 20x el tamaño del tablero
        
        # Loop detection: track last score change
        last_score = len(board.snake.body) + 1  # Initial snake length
        steps_since_food = 0
        max_steps_without_food = board_size * board_size * 2  # Must eat within this
        
        # Episode loop
        while True:
            step_count += 1
            
            # Handle quit event (only if visualizing)
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nTraining interrupted by user")
                        return stats
                
                board = game_ui.board
            
            # 1. Get State
            current_view = board.get_snake_view()
            prev_distance = board.get_distance_to_closest_food()
            
            # 2. Get Action (epsilon-greedy)
            action_idx = agent.get_action(current_view, epsilon)
            move_str = directions_map[action_idx]
            
            # 3. Execute action and get reward
            new_head, reward = board.move_snake(move_str)
            
            # Add distance-based reward shaping (encourage getting closer to food)
            if new_head is not None:  # Only if didn't die
                new_distance = board.get_distance_to_closest_food()
                distance_reward = (prev_distance - new_distance) * 0.2  # Small bonus for getting closer
                reward += distance_reward
            
            episode_reward += reward
            
            # Track score changes for loop detection
            current_score = len(board.snake.body) + 1
            if current_score != last_score:
                # Snake ate something (grew or shrank)
                steps_since_food = 0
                last_score = current_score
            else:
                steps_since_food += 1
            
            # Check if game is over
            done = False
            if new_head is None:  # Collision
                done = True
            
            # Loop detection: terminate if stuck going in circles
            if steps_since_food >= max_steps_without_food:
                done = True
                reward = -50  # Penalty for getting stuck
                episode_reward += reward
            
            # Timeout: absolute maximum steps
            if step_count >= max_steps:
                done = True
                reward = -50  # Penalty for extreme timeout
                episode_reward += reward
            
            # 4. Get next state
            next_view = board.get_snake_view() if not done else current_view
            
            # 5. Store experience and train the agent (Replay)
            agent.remember(current_view, action_idx, reward, next_view, done)
            loss = agent.replay()
            
            # 6. Update UI (if visualizing)
            if visualize:
                game_ui._update_ui()
                game_ui.clock.tick(fps)
            
            # Episode finished
            if done:
                final_score = len(board.snake.body) + 1
                
                # More detailed output for first 20 episodes
                if episode_num < 20 or debug:
                    print(f"Episode {start_episode + episode_num + 1} finished:")
                    print(f"  Steps: {step_count}")
                    print(f"  Final Score (snake length): {final_score}")
                    print(f"  Total Reward: {episode_reward:.1f}")
                    print(f"  Avg Reward per step: {episode_reward/step_count:.2f}")
                else:
                    print(f"Episode {start_episode + episode_num + 1} finished: Steps={step_count}, Score={final_score}, Reward={episode_reward:.1f}")
                
                # Update statistics
                stats['total_episodes'] += 1
                stats['total_score'] += final_score
                stats['scores'].append(final_score)
                if final_score > stats['high_score']:
                    stats['high_score'] = final_score
                    # Save best model checkpoint
                    best_model_path = f"models/best_snake_{board_size}x{board_size}_{num_episodes}ep.keras"
                    os.makedirs("models", exist_ok=True)
                    agent.model.save(best_model_path)
                    
                    # Save best model metadata
                    best_metadata = {
                        'board_size': board_size,
                        'total_episodes': start_episode + episode_num + 1,
                        'high_score': stats['high_score'],
                        'average_score': stats['total_score'] / (episode_num + 1), # Current average
                        'epsilon': epsilon,
                        'timestamp': datetime.now().isoformat()
                    }
                    best_metadata_path = best_model_path.replace('.keras', '_metadata.json')
                    with open(best_metadata_path, 'w') as f:
                        json.dump(best_metadata, f, indent=2)
                        
                    print(f"  🏆 New High Score! Best model saved to: {best_model_path}")
                
                # Decay epsilon (Linear Professional Schedule)
                # 1. Warmup: Keep at initial_epsilon for first 10% of sessions
                warmup_episodes = int(num_episodes * 0.10)
                # 2. Linear Decay: Fall to minimum at 60% of total sessions
                decay_end_episode = int(num_episodes * 0.70)
                epsilon_target = 0.005
                
                if episode_num < warmup_episodes:
                    epsilon = initial_epsilon
                elif episode_num < decay_end_episode:
                    # Linear interpolation from initial_epsilon to epsilon_target
                    progress = (episode_num - warmup_episodes) / (decay_end_episode - warmup_episodes)
                    epsilon = initial_epsilon - progress * (initial_epsilon - epsilon_target)
                else:
                    # Oscillation phase: spike to 0.02 every 5 episodes to avoid local minima
                    if (episode_num - decay_end_episode) % 1000 == 0:
                        epsilon = 0.02
                    else:
                        epsilon = epsilon_target
                break
        
        # Print progress every 10 episodes
        if (episode_num + 1) % 10 == 0:
            avg_score = stats['total_score'] / stats['total_episodes']
            last_10_avg = sum(stats['scores'][-10:]) / min(10, len(stats['scores']))
            print(f"\n--- Progress Report (Episode {episode_num + 1}) ---")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Last 10 Avg: {last_10_avg:.2f}")
            print(f"High Score: {stats['high_score']}")
            print(f"Epsilon: {epsilon:.3f}\n")
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"=== Training Complete ===")
    print(f"Total Episodes: {stats['total_episodes']}")
    print(f"Average Score: {stats['total_score'] / stats['total_episodes']:.2f}")
    print(f"High Score: {stats['high_score']}")
    print(f"Final Epsilon: {epsilon:.3f}")
    print(f"{'='*60}\n")
    
    return agent, stats, epsilon


def evaluate_agent(agent, board_size=10, num_games=10, visualize=True, fps=10):
    """
    Evaluate a trained agent (--dontlearn mode)
    """
    directions_map = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    print(f"\n{'='*60}")
    print(f"Evaluating agent: {num_games} games on {board_size}x{board_size} board")
    print(f"{'='*60}\n")
    
    scores = []
    
    for game_num in range(num_games):
        print(f"\n=== Game {game_num + 1}/{num_games} ===")
        
        if visualize:
            game_ui = BoardGameUI(board_size + 2)
            board = game_ui.board
        else:
            board = Board(board_size + 2)
            board.initialize()
        
        step_count = 0
        last_score = len(board.snake.body) + 1
        steps_since_food = 0
        max_steps_without_food = board_size * board_size
        max_steps = board_size * board_size * 10
        
        while True:
            step_count += 1
            
            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return scores
                board = game_ui.board
            
            # 1. Show vision
            current_view = board.get_snake_view()
            print_snake_vision(current_view, step_count, board_size)
            
            # 2. Algorithm decides
            action_idx = agent.get_action(current_view, epsilon=0)  # Pure exploitation
            move_str = directions_map[action_idx]
            print(f"Algorithm Decision -> [ {move_str} ]")
            print("="*30)
            # Execute action
            new_head, reward = board.move_snake(move_str)
            
            # Track score changes for loop detection
            current_score = len(board.snake.body) + 1
            if current_score != last_score:
                steps_since_food = 0
                last_score = current_score
            else:
                steps_since_food += 1
            
            if visualize:
                game_ui._update_ui()
                game_ui.clock.tick(fps)
            
            # Check game over (collision or loop)
            done = False
            if new_head is None or step_count >= max_steps or steps_since_food >= max_steps_without_food:
                done = True
                
            if done:
                final_score = len(board.snake.body) + 1
                scores.append(final_score)
                reason = "Collision" if new_head is None else "Loop/Timeout"
                print(f"Game {game_num + 1} finished ({reason}): Steps={step_count}, Score={final_score}")
                break
    
    # Print evaluation results
    print(f"\n{'='*60}")
    print(f"=== Evaluation Complete ===")
    print(f"Games Played: {len(scores)}")
    if scores:
        print(f"Average Score: {sum(scores) / len(scores):.2f}")
        print(f"Best Score: {max(scores)}")
        print(f"Worst Score: {min(scores)}")
    print(f"{'='*60}\n")
    
    return scores


def run_with_ui(agent, board_size=10):
    """
    Run with full UI (lobby, game, game over, stats)
    """
    pygame.init()
    
    # Show lobby to select board size
    lobby = LobbyScreen()
    selected_size = lobby.run()
    
    if selected_size is None:
        pygame.quit()
        return
    
    board_size = selected_size[0]
    print(f"\nStarting game with board size: {board_size}x{board_size}")
    
    # Play the game
    game_ui = BoardGameUI(board_size + 2)
    directions_map = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    step_count = 0
    last_score = len(game_ui.board.snake.body) + 1
    steps_since_food = 0
    max_steps_without_food = board_size * board_size
    max_steps = board_size * board_size * 10
    
    while True:
        step_count += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # 1. Show vision
        current_view = game_ui.board.get_snake_view()
        print_snake_vision(current_view, step_count, board_size)
        
        # 2. Algorithm decides
        action_idx = agent.get_action(current_view, epsilon=0)
        move_str = directions_map[action_idx]
        print(f"Algorithm Decision -> [ {move_str} ]")
        print("="*30)
        # Execute action
        new_head, reward = game_ui.board.move_snake(move_str)
        
        # Track score changes for loop detection
        current_score = len(game_ui.board.snake.body) + 1
        if current_score != last_score:
            steps_since_food = 0
            last_score = current_score
        else:
            steps_since_food += 1
            
        # Update UI
        game_ui._update_ui()
        game_ui.clock.tick(10)
        
        # Check game over (collision or loop)
        done = False
        reason = ""
        if new_head is None:
            done = True
            reason = "Collision"
        elif steps_since_food >= max_steps_without_food:
            done = True
            reason = "Stagnation (Loop)"
        elif step_count >= max_steps:
            done = True
            reason = "Hard Timeout"

        if done:
            final_score = len(game_ui.board.snake.body) + 1
            print(f"Game finished ({reason}): Steps={step_count}, Score={final_score}")
            
            # Show game over screen
            game_over_screen = GameOverScreen(final_score)
            choice = game_over_screen.run()
            
            if choice == 'new_game':
                # Restart
                game_ui = BoardGameUI(board_size + 2)
                step_count = 0
            elif choice == 'stats':
                stats = {
                    'total_games': 1,
                    'total_score': final_score,
                    'high_score': final_score,
                    'average_score': float(final_score)
                }
                stats_screen = StatsScreen(stats=stats)
                stats_screen.run()
                pygame.quit()
                return
            else:
                pygame.quit()
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or evaluate Snake AI agent')
    
    # Training arguments
    parser.add_argument('--sessions', type=int, default=100, 
                        help='Number of training episodes (default: 100)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load model from file (e.g., models/snake_model.keras)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save model to file (default: auto-generate name)')
    
    # Mode arguments
    parser.add_argument('--dontlearn', action='store_true',
                        help='Evaluation mode: do not train, only play')
    parser.add_argument('--visual', choices=['on', 'off'], default='off',
                        help='Enable visual interface (on/off)')
    
    # Game configuration
    parser.add_argument('--board-size', type=int, default=10, choices=[10, 14, 18],
                        help='Board size (10, 14, or 18)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for visualization (default: 10)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed prints')
    
    args = parser.parse_args()
    
    # Determine board size
    board_size = args.board_size
    
    # Load existing model if specified
    agent = None
    start_episode = 0
    loaded_metadata = None
    
    if args.load:
        print(f"Loading model from: {args.load}")
        # Use board_size 10 for the agent to match the trained model's input size (Option 1)
        agent = Agent(board_size=10)
        
        try:
            from tensorflow import keras
            agent.model = keras.models.load_model(args.load)
            print(f"✓ Model loaded successfully")
            
            # Try to load metadata
            metadata_path = args.load.replace('.keras', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    loaded_metadata = json.load(f)
                    start_episode = loaded_metadata.get('total_episodes', 0)
                    print(f"✓ Metadata loaded: {start_episode} episodes trained")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            agent = None
    
    # Mode selection
    if args.dontlearn:
        # Evaluation mode
        if agent is None:
            print("Error: --dontlearn requires --load <model_file>")
            sys.exit(1)
        
        if args.visual == 'on':
            # Full UI mode
            run_with_ui(agent, board_size)
        else:
            # Simple evaluation
            evaluate_agent(agent, board_size, num_games=args.sessions, visualize=False)
    
    else:
        # Training mode
        visualize = (args.visual == 'on')
        
        if visualize and agent is not None:
            print("Note: --visual on with --load will use simple visualization during training")
            print("      For full UI, use --visual on with --dontlearn")
        
        agent, stats, final_epsilon = train_agent(
            num_episodes=args.sessions,
            board_size=board_size,
            visualize=visualize,
            fps=args.fps,
            agent=agent,
            start_episode=start_episode,
            debug=args.debug
        )
        
        # Save model
        if args.save:
            model_path = args.save
        else:
            # Auto-generate filename
            total_episodes = start_episode + args.sessions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/snake_{board_size}x{board_size}_{total_episodes}ep_{timestamp}.keras"
        
        os.makedirs(os.path.dirname(model_path) or "models", exist_ok=True)
        agent.model.save(model_path)
        
        # Save metadata
        metadata = {
            'board_size': board_size,
            'total_episodes': start_episode + args.sessions,
            'high_score': stats['high_score'],
            'average_score': stats['total_score'] / stats['total_episodes'],
            'final_epsilon': final_epsilon,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = model_path.replace('.keras', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
    
    if args.visual == 'on':
        pygame.quit()
    
    print("\n✓ Program finished successfully")