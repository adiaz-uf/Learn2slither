"""
Main entry point for Learn2Slither.

Supports two modes:
  - Training mode (default): runs N episodes and saves the trained model.
  - Evaluation mode (--dontlearn): loads a model and plays without updates.

Board sizes: 10, 14, or 18 (playable area; walls add +2 internally).

Two training architectures, selected with --training-mode:
  - single (default): one model per board size, locked to --board-size.
  - multi: size-invariant 16-D features, episodes randomly sample 10/14/18
           so the same model can play any of those sizes (bonus part).
"""

import sys
import os
import pygame
import argparse
import json
import random
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from UI.game_ui import (  # noqa: E402
    BoardGameUI,
    GameOverScreen,
    StatsScreen,
    StartScreen,
)
from Agent import Agent  # noqa: E402
from Board import Board, INSTANT_GAMEOVER  # noqa: E402


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def print_snake_vision(view, step, board_size):
    """
    Print the snake's cross-shaped vision to the terminal.

    The layout mirrors the board: UP column above, HEAD in the middle,
    DOWN column below, with LEFT and RIGHT on the same row as the head.
    Each cell is represented by a single character (0, W, S, G, R, H).

    Args:
        view: list of 4 arms [UP, DOWN, LEFT, RIGHT], each a list of chars.
        step: current step number (displayed as a header).
        board_size: used only for display context.
    """
    clean = [[str(cell).strip()[-1] for cell in arm] for arm in view]
    up, down, left, right = clean

    left_str = "".join(reversed(left))
    right_str = "".join(right)
    indent = " " * len(left_str)

    print(f"\n--- Step {step} ---")
    for cell in reversed(up):
        print(f"{indent}{cell}")
    print(f"{left_str}H{right_str}")
    for cell in down:
        print(f"{indent}{cell}")


def _wait_for_step(visualize):
    """
    Block until the user confirms the next step.

    In terminal mode: waits for Enter.
    In visual mode: waits for any key press in the pygame window,
                    but still allows the window to be closed.
    Returns False if the user requests to quit, True otherwise.
    """
    if not visualize:
        input("Press Enter for next step...")
        return True

    # Pygame mode: drain the event queue until a KEYDOWN or QUIT event
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MULTI_BOARD_SIZES = [10, 14, 18]
MULTI_BOARD_WEIGHTS = [0.7, 0.15, 0.15]


def _sample_board_size(training_mode, default_size):
    """
    Pick the board size for the next episode.

    In single mode, always returns default_size. In multi mode, samples from
    [10, 14, 18] with weights [0.7, 0.15, 0.15] so 10x10 dominates training
    while the other sizes provide enough exposure for transfer.
    """
    if training_mode == 'multi':
        return random.choices(
            MULTI_BOARD_SIZES, weights=MULTI_BOARD_WEIGHTS, k=1
        )[0]
    return default_size


def train_agent(
    num_episodes=100,
    board_size=10,
    visualize=False,
    fps=10,
    agent=None,
    start_episode=0,
    debug=False,
    training_mode='single',
):
    """
    Train the DQN agent for a fixed number of episodes.

    Args:
        num_episodes: number of training episodes to run.
        board_size: playable board size (10, 14, or 18); walls add +2.
                    In multi mode this is only the fallback / display size;
                    each episode samples its own size from [10, 14, 18].
        visualize: if True, render each frame in a pygame window.
        fps: rendering speed when visualize is True.
        agent: pre-loaded Agent to continue training; new one if None.
        start_episode: episode offset used in logs when continuing training.
        debug: print detailed debug output for the first 2 episodes.
        training_mode: 'single' (one fixed board size) or 'multi' (sample per
                       episode from [10, 14, 18]).

    Returns:
        (agent, stats, final_epsilon)
    """
    if agent is None:
        if training_mode == 'multi':
            agent = Agent(board_size=None, mode='multi', debug=debug)
        else:
            agent = Agent(board_size=board_size, mode='single', debug=debug)
        initial_epsilon = 1.0
    else:
        agent.debug = debug
        # Low exploration when continuing from a trained model
        initial_epsilon = 0.005

    epsilon = initial_epsilon
    directions_map = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    stats = {
        'total_episodes': 0,
        'total_score': 0,
        'high_score': 0,
        'best_rolling_avg': 0.0,
        'scores': [],
    }

    print(f"\n{'=' * 60}")
    if start_episode > 0:
        print(f"Continuing training from episode {start_episode + 1}")
    if training_mode == 'multi':
        weights_pct = [int(w * 100) for w in MULTI_BOARD_WEIGHTS]
        sizes_str = ", ".join(
            f"{s}x{s} ({w}%)"
            for s, w in zip(MULTI_BOARD_SIZES, weights_pct)
        )
        print(
            f"Training (multi-size): {num_episodes} episodes — {sizes_str}"
        )
    else:
        print(
            f"Training: {num_episodes} episodes on "
            f"{board_size}x{board_size} board"
        )
    if debug:
        print("DEBUG MODE ENABLED")
    print(f"{'=' * 60}\n")

    # Epsilon schedule parameters.
    # Fixed 200-episode warmup. Decay covers 35% of the remaining episodes
    # (capped at 4500) so that runs of 5k-20k all get ~30-37% learning time
    # instead of rushing into exploitation too early.
    warmup_episodes = 200
    decay_end_episode = 200 + min(int((num_episodes - 200) * 0.35), 4500)
    epsilon_target = 0.01

    for episode_num in range(num_episodes):
        # In multi mode, the board size for this episode is sampled here
        current_board_size = _sample_board_size(training_mode, board_size)

        if training_mode == 'multi':
            print(f"\n=== Episode {start_episode + episode_num + 1}"
                  f" | Epsilon: {epsilon:.3f}"
                  f" | Board: {current_board_size}x{current_board_size} ===")
        else:
            print(f"\n=== Episode {start_episode + episode_num + 1}"
                  f" | Epsilon: {epsilon:.3f} ===")

        # Enable detailed debug output for the first 2 episodes only
        agent.debug = debug and episode_num < 2

        # Set up the game environment
        if visualize:
            game_ui = BoardGameUI(current_board_size + 2)
            board = game_ui.board
        else:
            board = Board(current_board_size + 2)
            board.initialize()

        step_count = 0
        episode_reward = 0.0
        max_steps = current_board_size * current_board_size * 20

        # Loop-detection: track steps elapsed without any food event.
        # max_steps_without_food is recomputed each step inside the loop
        # so that longer snakes get proportionally more time to navigate.
        last_score = len(board.snake.body) + 1
        steps_since_food = 0

        while True:
            step_count += 1

            if visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nTraining interrupted by user.")
                        return agent, stats, epsilon
                board = game_ui.board

            # 1. Observe current state
            current_view = board.get_snake_view()

            # 2. Select action (epsilon-greedy)
            action_idx = agent.get_action(current_view, epsilon)
            move_str = directions_map[action_idx]

            # 3. Execute action and receive reward
            new_head, reward = board.move_snake(move_str)

            episode_reward += reward

            # Update loop-detection counter
            current_score = len(board.snake.body) + 1
            if current_score != last_score:
                steps_since_food = 0
                last_score = current_score
            else:
                steps_since_food += 1

            # Determine if the episode is over
            done = new_head is None

            # Recompute the food-timeout threshold using the current snake
            # length. Longer snakes occupy more cells and need more steps to
            # reach food without being falsely terminated by the loop detector.
            #   length 12 -> 200 steps  (board_size^2 * 2)
            #   length 18 -> 300 steps
            #   length 24 -> 400 steps
            #   length 30 -> 500 steps
            #   length 36 -> 600 steps
            #   length 42 -> 700 steps
            #   length 48 -> 800 steps
            current_length = len(board.snake.body) + 1
            max_steps_without_food = (
                current_board_size * current_board_size
                * max(2, current_length // 6)
            )

            if steps_since_food >= max_steps_without_food:
                done = True
                # Loop penalty matches the death penalty
                reward = INSTANT_GAMEOVER
                episode_reward += reward

            if step_count >= max_steps:
                done = True
                reward = -50  # Penalty for exceeding the step limit
                episode_reward += reward

            # 4. Observe next state
            next_view = board.get_snake_view() if not done else current_view

            # 5. Store transition and train
            agent.remember(current_view, action_idx, reward, next_view, done)
            agent.replay()

            # 6. Render frame
            if visualize:
                game_ui._update_ui()
                game_ui.clock.tick(fps)

            if done:
                final_score = len(board.snake.body) + 1

                ep_id = start_episode + episode_num + 1
                if episode_num < 20 or debug:
                    print(f"Episode {ep_id} finished:")
                    print(f"  Steps: {step_count}")
                    print(f"  Final Score: {final_score}")
                    print(f"  Total Reward: {episode_reward:.1f}")
                    avg = episode_reward / step_count
                    print(f"  Avg Reward/step: {avg:.2f}")
                else:
                    print(
                        f"Episode {ep_id} finished:"
                        f" Steps={step_count},"
                        f" Score={final_score},"
                        f" Reward={episode_reward:.1f}"
                    )

                stats['total_episodes'] += 1
                stats['total_score'] += final_score
                stats['scores'].append(final_score)

                # Output paths depend on training mode:
                #   single -> models/<size>x<size>/...
                #   multi  -> models/multi/...
                if training_mode == 'multi':
                    model_dir = "models/multi"
                    name_tag = "multi"
                else:
                    model_dir = f"models/{board_size}x{board_size}"
                    name_tag = f"{board_size}x{board_size}"

                if final_score > stats['high_score']:
                    stats['high_score'] = final_score
                    # Checkpoint: save the best model seen so far
                    best_path = (f"{model_dir}/best_snake_{name_tag}"
                                 f"_{num_episodes}ep.keras")
                    os.makedirs(model_dir, exist_ok=True)
                    agent.model.save(best_path)
                    best_meta = {
                        'mode': training_mode,
                        'board_size': (None if training_mode == 'multi'
                                       else board_size),
                        'total_episodes': start_episode + episode_num + 1,
                        'high_score': stats['high_score'],
                        'average_score': (
                            stats['total_score'] / (episode_num + 1)
                        ),
                        'epsilon': epsilon,
                        'timestamp': datetime.now().isoformat(),
                    }
                    meta_path = best_path.replace(
                        '.keras', '_metadata.json'
                    )
                    with open(meta_path, 'w') as f:
                        json.dump(best_meta, f, indent=2)
                    print(
                        f"  New high score! "
                        f"Best model saved to: {best_path}"
                    )

                # Save model whenever the 200-episode rolling average
                # improves, regardless of whether a new single-game high
                # score was set. This captures policy improvements that
                # never produce a lucky peak.
                rolling_window = stats['scores'][-200:]
                rolling_200 = sum(rolling_window) / max(1, len(rolling_window))
                if rolling_200 > stats['best_rolling_avg']:
                    stats['best_rolling_avg'] = rolling_200
                    avg_path = (
                        f"{model_dir}/best_avg_snake_{name_tag}"
                        f"_{num_episodes}ep.keras"
                    )
                    os.makedirs(model_dir, exist_ok=True)
                    agent.model.save(avg_path)
                    avg_meta = {
                        'mode': training_mode,
                        'board_size': (None if training_mode == 'multi'
                                       else board_size),
                        'total_episodes': start_episode + episode_num + 1,
                        'high_score': stats['high_score'],
                        'rolling_200_avg': rolling_200,
                        'epsilon': epsilon,
                        'timestamp': datetime.now().isoformat(),
                    }
                    avg_meta_path = avg_path.replace(
                        '.keras', '_metadata.json'
                    )
                    with open(avg_meta_path, 'w') as f:
                        json.dump(avg_meta, f, indent=2)
                    print(
                        f"  New best rolling avg ({rolling_200:.2f})! "
                        f"Avg model saved to: {avg_path}"
                    )

                break  # End of episode

        # Decay epsilon according to a three-phase schedule:
        # 1. Warmup (first 10%): keep epsilon at its initial value.
        # 2. Linear decay (10% -> 70%): decrease to epsilon_target.
        # 3. Exploitation (70% -> 100%): hold at epsilon_target with
        #    occasional spikes.
        if episode_num < warmup_episodes:
            epsilon = initial_epsilon
        elif episode_num < decay_end_episode:
            progress = ((episode_num - warmup_episodes)
                        / (decay_end_episode - warmup_episodes))
            epsilon = (
                initial_epsilon
                - progress * (initial_epsilon - epsilon_target)
            )
        else:
            # Exploration cycles in the exploitation phase: for the first
            # 30 of every 200 episodes, raise epsilon to 0.05 AND spike
            # the learning rate to 0.001 so the network can actually
            # absorb the novel transitions generated by exploration (at
            # base LR they are diluted by the 150K memory of greedy
            # experiences and have negligible gradient effect).
            cycle_pos = (episode_num - decay_end_episode) % 200
            if cycle_pos < 30:
                epsilon = 0.05
                agent.set_learning_rate(0.001)
            else:
                epsilon = epsilon_target
                agent.set_learning_rate(0.0005)

        # Progress summary every 10 episodes
        if (episode_num + 1) % 10 == 0:
            avg = stats['total_score'] / stats['total_episodes']
            last_10 = (
                sum(stats['scores'][-10:])
                / min(10, len(stats['scores']))
            )
            print(f"\n--- Progress (Episode {episode_num + 1}) ---")
            print(f"  Average Score:    {avg:.2f}")
            print(f"  Last 10 Average:  {last_10:.2f}")
            print(f"  High Score:       {stats['high_score']}")
            print(f"  Epsilon:          {epsilon:.3f}\n")

    print(f"\n{'=' * 60}")
    print("Training complete.")
    print(f"  Total Episodes: {stats['total_episodes']}")
    final_avg = stats['total_score'] / stats['total_episodes']
    print(f"  Average Score:  {final_avg:.2f}")
    print(f"  High Score:     {stats['high_score']}")
    print(f"  Final Epsilon:  {epsilon:.3f}")
    print(f"{'=' * 60}\n")

    return agent, stats, epsilon


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent,
    board_size=10,
    num_games=10,
    visualize=False,
    fps=10,
    step_by_step=False,
):
    """
    Evaluate a trained agent without updating its weights.

    Prints vision and action for every step. Optionally pauses between steps.

    Args:
        agent: trained Agent instance (epsilon=0, pure exploitation).
        board_size: playable board size.
        num_games: number of evaluation games to run.
        visualize: if True, render in a pygame window.
        fps: rendering speed when visualize is True.
        step_by_step: if True, pause after each step until the user advances.

    Returns:
        List of final scores, one per game.
    """
    directions_map = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {num_games} games on {board_size}x{board_size} board")
    print(f"{'=' * 60}\n")

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

            # Display vision and action in the terminal
            current_view = board.get_snake_view()
            print_snake_vision(current_view, step_count, board_size)

            action_idx = agent.get_action(current_view, epsilon=0)
            move_str = directions_map[action_idx]
            print(f"Action -> [ {move_str} ]")
            print("-" * 30)

            # Step-by-step: pause until user advances
            if step_by_step:
                if not _wait_for_step(visualize):
                    return scores

            new_head, _ = board.move_snake(move_str)

            current_score = len(board.snake.body) + 1
            if current_score != last_score:
                steps_since_food = 0
                last_score = current_score
            else:
                steps_since_food += 1

            if visualize:
                game_ui._update_ui()
                game_ui.clock.tick(fps)

            done = (new_head is None
                    or step_count >= max_steps
                    or steps_since_food >= max_steps_without_food)

            if done:
                final_score = len(board.snake.body) + 1
                scores.append(final_score)
                reason = "Collision" if new_head is None else "Loop/Timeout"
                print(f"Game {game_num + 1} finished ({reason}):"
                      f" Steps={step_count}, Score={final_score}")
                break

    print(f"\n{'=' * 60}")
    print("Evaluation complete.")
    print(f"  Games Played:   {len(scores)}")
    if scores:
        print(f"  Average Score:  {sum(scores) / len(scores):.2f}")
        print(f"  Best Score:     {max(scores)}")
        print(f"  Worst Score:    {min(scores)}")
    print(f"{'=' * 60}\n")

    return scores


# ---------------------------------------------------------------------------
# Full UI run (visual evaluation with lobby, game-over, stats screens)
# ---------------------------------------------------------------------------

def run_with_ui(agent, board_size, step_by_step=False, fps=10):
    """
    Run an interactive evaluation session with the full pygame UI.

    The StartScreen lets the user pick a board size; that size is used
    for the rest of the session. For 'single'-mode models the selector
    is restricted to the size the model was trained for. For 'multi'-mode
    models all sizes are offered (size-portability bonus).

    Args:
        agent: trained Agent instance.
        board_size: fallback board size (used as the default selection;
                    in single mode it is the only allowed size).
        step_by_step: if True, pause after each game step.
        fps: rendering speed (frames per second).
    """
    pygame.init()
    directions_map = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    # Cumulative statistics for the entire session
    session_stats = {
        'total_games': 0,
        'total_score': 0,
        'high_score': 0,
        'average_score': 0.0,
    }

    # Restrict the selector when the model is locked to a single size
    if agent.mode == 'single':
        allowed_sizes = [agent.board_size]
    else:
        allowed_sizes = StartScreen.BOARD_SIZES

    # Show start screen and let the user pick the board size
    start_screen = StartScreen(allowed_sizes=allowed_sizes)
    action, selected_size = start_screen.run()
    if action == 'quit':
        pygame.quit()
        return

    board_size = selected_size

    while True:

        # Initialise a fresh game
        game_ui = BoardGameUI(board_size + 2)
        step_count = 0
        last_score = len(game_ui.board.snake.body) + 1
        steps_since_food = 0
        max_steps_without_food = board_size * board_size
        max_steps = board_size * board_size * 10

        # Game loop
        while True:
            step_count += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            current_view = game_ui.board.get_snake_view()
            print_snake_vision(current_view, step_count, board_size)

            action_idx = agent.get_action(current_view, epsilon=0)
            move_str = directions_map[action_idx]
            print(f"Action -> [ {move_str} ]")
            print("-" * 30)

            # Step-by-step: wait for a key press before proceeding
            if step_by_step:
                if not _wait_for_step(visualize=True):
                    pygame.quit()
                    return

            new_head, _ = game_ui.board.move_snake(move_str)

            current_score = len(game_ui.board.snake.body) + 1
            if current_score != last_score:
                steps_since_food = 0
                last_score = current_score
            else:
                steps_since_food += 1

            game_ui._update_ui()
            game_ui.clock.tick(fps)

            done = False
            reason = ""
            if new_head is None:
                done = True
                reason = "Collision"
            elif steps_since_food >= max_steps_without_food:
                done = True
                reason = "Stagnation"
            elif step_count >= max_steps:
                done = True
                reason = "Timeout"

            if done:
                final_score = len(game_ui.board.snake.body) + 1
                print(f"Game finished ({reason}):"
                      f" Steps={step_count}, Score={final_score}")

                # Update cumulative session statistics
                session_stats['total_games'] += 1
                session_stats['total_score'] += final_score
                if final_score > session_stats['high_score']:
                    session_stats['high_score'] = final_score
                session_stats['average_score'] = (
                    session_stats['total_score'] / session_stats['total_games']
                )

                # Show game-over screen
                game_over = GameOverScreen(final_score)
                choice = game_over.run()

                if choice == 'new_game':
                    break  # Restart the outer while loop (new game)
                elif choice == 'stats':
                    stats_screen = StatsScreen(stats=session_stats)
                    stats_screen.run()
                    pygame.quit()
                    return
                else:  # 'quit'
                    pygame.quit()
                    return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train or evaluate the Snake DQN agent.'
    )

    # --- Training arguments ---
    parser.add_argument(
        '--sessions', type=int, default=100,
        help='Number of training episodes (default: 100)',
    )
    parser.add_argument(
        '--load', type=str, default=None,
        help='Path to a pre-trained model file (.keras)',
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Path to save the trained model (auto-generated if omitted)',
    )

    # --- Mode arguments ---
    parser.add_argument(
        '--dontlearn', action='store_true',
        help='Evaluation mode: run the agent without updating weights',
    )
    parser.add_argument(
        '--visual', choices=['on', 'off'], default='off',
        help='Enable pygame visualization (default: off)',
    )
    parser.add_argument(
        '--step-by-step', action='store_true',
        help=('Pause after every step and wait for user input '
              '(evaluation only)'),
    )

    # --- Game configuration ---
    parser.add_argument(
        '--board-size', type=int, default=10, choices=[10, 14, 18],
        help=('Playable board size: 10, 14, or 18 (default: 10). '
              'In multi training mode this is only the fallback size.'),
    )
    parser.add_argument(
        '--training-mode', choices=['single', 'multi'], default='single',
        help="'single': train one model dedicated to --board-size (default). "
             "'multi': size-invariant training, each episode samples a board "
             "size from [10, 14, 18] (size-portability bonus).",
    )
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Frames per second during visualization (default: 10)',
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Print detailed debug output for the first two training episodes',
    )

    args = parser.parse_args()

    board_size = args.board_size
    training_mode = args.training_mode
    agent = None
    start_episode = 0

    # --- Load model if requested ---
    if args.load:
        metadata_path = args.load.replace('.keras', '_metadata.json')
        loaded_board_size = None
        loaded_mode = None

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            loaded_board_size = metadata.get('board_size')
            loaded_mode = metadata.get('mode', 'single')  # legacy default
            start_episode = metadata.get('total_episodes', 0)
        else:
            print(
                f"Warning: no metadata found for '{args.load}'. "
                f"Assuming mode='single' and --board-size {board_size}."
            )
            loaded_mode = 'single'

        # When continuing training, the loaded model's mode wins over the CLI
        # flag — the architectures are incompatible and would crash on load.
        if loaded_mode != training_mode:
            print(
                f"Note: loaded model was trained in '{loaded_mode}' mode; "
                f"using that instead of --training-mode {training_mode}."
            )
            training_mode = loaded_mode

        # In single mode, the board size is part of the architecture
        # (input shape depends on it). In multi mode any size works.
        if loaded_mode == 'single':
            if (loaded_board_size is not None
                    and loaded_board_size != board_size):
                print(
                    f"Note: this single-mode model was trained for "
                    f"{loaded_board_size}x{loaded_board_size}; "
                    f"overriding --board-size to that value."
                )
                board_size = loaded_board_size
            agent = Agent(board_size=board_size, mode='single')
        else:  # multi
            agent = Agent(board_size=None, mode='multi')

        try:
            from tensorflow import keras as tf_keras
            agent.model = tf_keras.models.load_model(args.load)
            agent.update_target_model()
            print(f"Model loaded: {args.load} (mode={loaded_mode})")
            if start_episode:
                print(f"Resuming from episode {start_episode}")
        except Exception as exc:
            print(f"Error loading model: {exc}")
            sys.exit(1)

    # --- Mode: evaluation (--dontlearn) ---
    if args.dontlearn:
        if agent is None:
            print("Error: --dontlearn requires --load <model_path>.")
            sys.exit(1)

        if args.visual == 'on':
            pygame.init()
            run_with_ui(
                agent, board_size,
                step_by_step=args.step_by_step,
                fps=args.fps,
            )
        else:
            evaluate_agent(
                agent,
                board_size=board_size,
                num_games=args.sessions,
                visualize=False,
                step_by_step=args.step_by_step,
            )

    # --- Mode: training ---
    else:
        visualize = args.visual == 'on'

        agent, stats, final_epsilon = train_agent(
            num_episodes=args.sessions,
            board_size=board_size,
            visualize=visualize,
            fps=args.fps,
            agent=agent,
            start_episode=start_episode,
            debug=args.debug,
            training_mode=training_mode,
        )

        # Determine output path
        if args.save:
            model_path = args.save
        else:
            total_ep = start_episode + args.sessions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if training_mode == 'multi':
                model_dir = "models/multi"
                model_path = (
                    f"{model_dir}/snake_multi_{total_ep}ep_{timestamp}.keras"
                )
            else:
                model_dir = f"models/{board_size}x{board_size}"
                model_path = (
                    f"{model_dir}/snake_{board_size}x{board_size}"
                    f"_{total_ep}ep_{timestamp}.keras"
                )

        os.makedirs(os.path.dirname(model_path) or "models", exist_ok=True)
        agent.model.save(model_path)

        metadata = {
            'mode': training_mode,
            'board_size': (None if training_mode == 'multi' else board_size),
            'total_episodes': start_episode + args.sessions,
            'high_score': stats['high_score'],
            'average_score': stats['total_score'] / stats['total_episodes'],
            'final_epsilon': final_epsilon,
            'timestamp': datetime.now().isoformat(),
        }
        metadata_path = model_path.replace('.keras', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved:    {model_path}")
        print(f"Metadata saved: {metadata_path}")

    if args.visual == 'on':
        pygame.quit()

    print("\nProgram finished successfully.")
