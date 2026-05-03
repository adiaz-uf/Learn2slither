"""
Hyperparameter tuning with Optuna.

Runs many short training trials with different hyperparameter combinations
and reports the best ones. The trial metric is the rolling-200 average of
the final score, computed from the last 200 episodes of each trial.

Tunable hyperparameters (matches the user's spec):
  - learning_rate : log-uniform [1e-4, 2e-3]
  - gamma         : uniform [0.92, 0.99]
  - batch_size    : categorical {128, 256, 512, 1024}
  - NO_EAT        : uniform [-2.0, -0.1]
  - hidden_layers : 2-4 layers, 64-512 units each (step 64)

Pruning:
  Optuna's MedianPruner cuts unpromising trials early using the
  intermediate rolling-200 averages reported by train_agent every 100
  episodes. Trials whose rolling-200 lags significantly behind the median
  of completed trials are aborted.

Usage:
    python src/AI_Model/tune.py --trials 50 --episodes-per-trial 1500
    python src/AI_Model/tune.py --training-mode multi --trials 30 \\
        --episodes-per-trial 1000

Storage:
    Results are persisted in an SQLite study so runs can be resumed and
    inspected. Default path: optuna_studies/<study_name>.db
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Quiet TF warnings unless TF_LOG is explicitly set
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import optuna  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402

# Make sibling modules importable when running this file directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import train_agent  # noqa: E402


def _suggest_hyperparameters(trial, mode):
    """Sample a complete hyperparameter set from the trial's search space."""
    learning_rate = trial.suggest_float(
        'learning_rate', 1e-4, 2e-3, log=True
    )
    gamma = trial.suggest_float('gamma', 0.92, 0.99)
    batch_size = trial.suggest_categorical(
        'batch_size', [128, 256, 512, 1024]
    )
    no_eat = trial.suggest_float('no_eat', -2.0, -0.1)

    # Architecture: 2-4 hidden layers, 64-512 units each.
    # Conditional sampling — only sample as many layer widths as needed.
    num_layers = trial.suggest_int('num_layers', 2, 4)
    hidden_layers = [
        trial.suggest_int(f'units_l{i}', 64, 512, step=64)
        for i in range(num_layers)
    ]

    return {
        'learning_rate': learning_rate,
        'gamma': gamma,
        'batch_size': batch_size,
        'no_eat': no_eat,
        'hidden_layers': hidden_layers,
        'num_layers': num_layers,
        'mode': mode,
    }


def _build_objective(args):
    def objective(trial):
        hp = _suggest_hyperparameters(trial, args.training_mode)

        # Pruning callback: train_agent calls this every 100 episodes
        # with the current rolling-200 average. We forward it to Optuna
        # which decides whether to prune.
        def callback(episode, rolling_avg):
            trial.report(rolling_avg, step=episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        rewards_override = {'NO_EAT': hp['no_eat']}

        try:
            _, stats, _ = train_agent(
                num_episodes=args.episodes_per_trial,
                board_size=args.board_size,
                training_mode=args.training_mode,
                visualize=False,
                learning_rate=hp['learning_rate'],
                gamma=hp['gamma'],
                batch_size=hp['batch_size'],
                hidden_layers=hp['hidden_layers'],
                rewards=rewards_override,
                intermediate_callback=callback,
                save_checkpoints=False,
                verbose=False,
            )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            # Surface the failure to Optuna instead of crashing the study
            print(f"Trial {trial.number} failed: {exc}")
            raise optuna.exceptions.TrialPruned()

        scores = stats['scores']
        if len(scores) >= 200:
            metric = sum(scores[-200:]) / 200
        else:
            metric = sum(scores) / max(1, len(scores))

        # Store some auxiliary info on the trial for analysis later
        trial.set_user_attr('high_score', stats['high_score'])
        trial.set_user_attr('episodes_completed', len(scores))

        return metric

    return objective


def _summarize(study):
    print(f"\n{'=' * 60}")
    print("Optimization complete.")
    print(f"  Trials run:    {len(study.trials)}")
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    failed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
    ]
    print(f"  Completed:     {len(completed)}")
    print(f"  Pruned:        {len(pruned)}")
    print(f"  Failed:        {len(failed)}")

    if not completed:
        print("\nNo trial completed; nothing to report.")
        return None

    print(f"\n  Best rolling-200 avg: {study.best_value:.3f}")
    print("  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"{'=' * 60}\n")
    return study.best_params


def _save_results(args, study, best_params):
    out_dir = 'optuna_results'
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = (
        f"{out_dir}/best_{args.training_mode}_{timestamp}.json"
    )
    payload = {
        'study_name': args.study_name,
        'training_mode': args.training_mode,
        'board_size': args.board_size,
        'episodes_per_trial': args.episodes_per_trial,
        'n_trials_completed': len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]),
        'best_value': float(study.best_value) if best_params else None,
        'best_params': best_params,
        'timestamp': timestamp,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to: {out_path}")

    # Also print a ready-to-paste training command for the best params
    if best_params:
        hidden_units = [
            best_params[f'units_l{i}']
            for i in range(best_params['num_layers'])
        ]
        hidden_arg = ','.join(str(u) for u in hidden_units)
        cmd = [
            "python src/AI_Model/main.py",
            "--sessions 10000",
            f"--training-mode {args.training_mode}",
        ]
        if args.training_mode == 'single':
            cmd.append(f"--board-size {args.board_size}")
        cmd.extend([
            f"--learning-rate {best_params['learning_rate']:.6f}",
            f"--gamma {best_params['gamma']:.4f}",
            f"--batch-size {best_params['batch_size']}",
            f"--no-eat-reward {best_params['no_eat']:.3f}",
            f"--hidden-layers {hidden_arg}",
        ])
        print("\nTo train a full run with these params:")
        print("    " + " \\\n        ".join(cmd))


def main():
    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter tuning for Snake DQN.'
    )
    parser.add_argument(
        '--trials', type=int, default=50,
        help='Number of Optuna trials to run (default: 50).',
    )
    parser.add_argument(
        '--episodes-per-trial', type=int, default=1500,
        help='Episodes to train per trial (default: 1500).',
    )
    parser.add_argument(
        '--training-mode', choices=['single', 'multi'], default='single',
        help='DQN training mode (default: single).',
    )
    parser.add_argument(
        '--board-size', type=int, default=10, choices=[10, 14, 18],
        help='Board size for single mode (default: 10).',
    )
    parser.add_argument(
        '--study-name', type=str, default=None,
        help='Optuna study name (default: snake-<mode>-<board>).',
    )
    parser.add_argument(
        '--storage', type=str, default=None,
        help=(
            'Optuna storage URL '
            '(default: sqlite:///optuna_studies/<study>.db).'
        ),
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='TPE sampler seed for reproducible search (default: 42).',
    )
    parser.add_argument(
        '--n-startup-trials', type=int, default=5,
        help='Random trials before TPE kicks in (default: 5).',
    )
    parser.add_argument(
        '--n-warmup-steps', type=int, default=200,
        help=(
            'Episodes before pruning becomes active for a trial '
            '(default: 200).'
        ),
    )
    args = parser.parse_args()

    # Default study name
    if args.study_name is None:
        if args.training_mode == 'multi':
            args.study_name = 'snake-multi'
        else:
            args.study_name = f'snake-single-{args.board_size}'

    # Default SQLite storage
    if args.storage is None:
        os.makedirs('optuna_studies', exist_ok=True)
        args.storage = f'sqlite:///optuna_studies/{args.study_name}.db'

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.n_warmup_steps,
        interval_steps=100,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"{'=' * 60}")
    print(f"  Study:              {args.study_name}")
    print(f"  Storage:            {args.storage}")
    print(f"  Trials this run:    {args.trials}")
    print(f"  Episodes / trial:   {args.episodes_per_trial}")
    print(f"  Mode:               {args.training_mode}")
    if args.training_mode == 'single':
        print(f"  Board:              {args.board_size}x{args.board_size}")
    print(f"  Existing trials:    {len(study.trials)}")
    print(f"{'=' * 60}\n")

    objective = _build_objective(args)
    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            show_progress_bar=False,
            catch=(),
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user — partial results below.")

    best_params = _summarize(study)
    _save_results(args, study, best_params)


if __name__ == '__main__':
    main()
