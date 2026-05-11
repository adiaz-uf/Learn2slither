"""
Hyperparameter tuning with Optuna.

Runs many short training trials with different hyperparameter combinations
and reports the best ones. The trial metric is the rolling-200 average of
the final score, computed from the last 200 episodes of each trial.

Tunable hyperparameters (centred on values that work for the 16-D
feature input + a small MLP):
  - learning_rate : log-uniform [3e-4, 1.5e-3]
  - gamma         : uniform [0.95, 0.99]
  - batch_size    : categorical {128, 256, 512, 1024}
  - NO_EAT        : uniform [-1.0, -0.2]
  - hidden_layers : 2-3 layers, 32-160 units each (step 32)

The first trial of every fresh study is seeded with a known-good
baseline (lr=5e-4, gamma=0.98, NO_EAT=-0.3, [64,64], batch=512) so the
study has a reasonable reference value from the start instead of
suffering through random startup samples.

Pruning (lenient by design — late-bloomer-friendly):
  - Pruner: PercentilePruner(25) wrapped by PatientPruner(patience=2).
    Only the bottom 25% of trials get pruned at any checkpoint, and a
    trial is allowed two consecutive bad readings before pruning fires.
    The combination avoids killing trials that improve slowly or that
    have a temporary dip in the rolling-200 average.
  - Warmup: pruning is disabled until ``n_warmup_steps`` (default 1500)
    so trials always get to finish their epsilon-decay phase before
    being judged. With the default schedule, decay ends near episode
    1880 for a 5000-episode trial, so anything after step 1500 is
    showing genuine policy quality, not exploration noise.
  - Reporting: train_agent calls back every 100 episodes with the
    rolling-200 average of episode scores.
  - Use ``--pruner none`` to disable pruning entirely (max safety).
  - Use ``--pruner median`` for the classic (more aggressive) behaviour.

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
from optuna.pruners import (  # noqa: E402
    MedianPruner,
    NopPruner,
    PatientPruner,
    PercentilePruner,
)
from optuna.samplers import TPESampler  # noqa: E402

# Make sibling modules importable when running this file directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import train_agent  # noqa: E402


def _suggest_hyperparameters(trial, mode):
    """
    Sample a complete hyperparameter set from the trial's search space.

    Ranges are tuned for a 16-D dense input + a small MLP: layer widths
    above ~160 over-parameterise the network, lr below 3e-4 trains too
    slowly within a single trial's budget, and NO_EAT below -1.0 makes
    the agent prefer suicide over exploration. The baseline known-good
    config (5e-4 / 0.98 / -0.3 / [64,64]) sits in the middle of these
    ranges so TPE can refine around it.
    """
    learning_rate = trial.suggest_float(
        'learning_rate', 3e-4, 1.5e-3, log=True
    )
    gamma = trial.suggest_float('gamma', 0.95, 0.99)
    batch_size = trial.suggest_categorical(
        'batch_size', [128, 256, 512, 1024]
    )
    no_eat = trial.suggest_float('no_eat', -1.0, -0.2)

    # Architecture: 2-3 hidden layers, 32-160 units each.
    num_layers = trial.suggest_int('num_layers', 2, 3)
    hidden_layers = [
        trial.suggest_int(f'units_l{i}', 32, 160, step=32)
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


# Known-good hyperparameter set that produced solid scores in earlier
# multi-mode training. Used as the first enqueued trial so every fresh
# study starts from a sensible reference instead of pure random samples.
BASELINE_PARAMS = {
    'learning_rate': 5e-4,
    'gamma': 0.98,
    'batch_size': 512,
    'no_eat': -0.3,
    'num_layers': 2,
    'units_l0': 64,
    'units_l1': 64,
}


def _enqueue_baseline(study):
    """
    Enqueue the known-good baseline as the next trial (if not already
    present). Subsequent calls are idempotent: re-running ``make tune``
    on an existing study won't duplicate the baseline because the
    matching params will be skipped by Optuna.
    """
    # Check if any completed trial already has these exact params.
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            if all(
                t.params.get(k) == v for k, v in BASELINE_PARAMS.items()
            ):
                return False
    study.enqueue_trial(BASELINE_PARAMS)
    return True


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
        help=(
            'Number of completed trials required before any pruning '
            'comparison runs (default: 5). Below this threshold every '
            'trial finishes regardless of intermediate values.'
        ),
    )
    parser.add_argument(
        '--n-warmup-steps', type=int, default=1500,
        help=(
            'Episodes a trial must run before becoming eligible for '
            'pruning (default: 1500). Set high enough that the '
            'epsilon-decay phase has finished — episodes before this '
            'are dominated by exploration noise.'
        ),
    )
    parser.add_argument(
        '--pruner', choices=['percentile', 'median', 'none'],
        default='percentile',
        help=(
            "Pruning strategy. 'percentile' (default) only kills the "
            "bottom 25%% of trials at each checkpoint; 'median' is the "
            "classic (more aggressive) Optuna default; 'none' disables "
            "pruning entirely."
        ),
    )
    parser.add_argument(
        '--prune-percentile', type=float, default=25.0,
        help=(
            "Percentile threshold for the 'percentile' pruner: trials "
            "whose intermediate value falls below this percentile of "
            "completed trials at the same step are pruned. Lower = "
            "more lenient (default: 25.0)."
        ),
    )
    parser.add_argument(
        '--patience', type=int, default=2,
        help=(
            'Wrap the pruner with PatientPruner(patience=N): a trial '
            "must report at least N consecutive bad values before being "
            'pruned (default: 2). Set to 0 to disable patience.'
        ),
    )
    parser.add_argument(
        '--no-baseline', action='store_true',
        help=(
            'Do not enqueue the known-good baseline as the first trial. '
            'Off by default (the baseline is always enqueued on a fresh '
            'study; idempotent on resume).'
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
    # Build the base pruner from --pruner. PercentilePruner with a low
    # percentile (default 25) is the lenient choice — it only prunes
    # trials that fall below the bottom-25% of completed trials at the
    # same step, leaving median-or-better trials alive.
    if args.pruner == 'none':
        base_pruner = NopPruner()
    elif args.pruner == 'median':
        base_pruner = MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
            interval_steps=100,
        )
    else:  # 'percentile' (default)
        base_pruner = PercentilePruner(
            percentile=args.prune_percentile,
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
            interval_steps=100,
        )

    # PatientPruner adds hysteresis: a trial that drops below the
    # threshold once is given a few more checkpoints to recover before
    # being killed. Avoids pruning trials with transient dips in the
    # rolling-200 average (especially common during the first exploration
    # cycle after the decay phase ends).
    if args.patience > 0 and args.pruner != 'none':
        pruner = PatientPruner(base_pruner, patience=args.patience)
    else:
        pruner = base_pruner

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Seed the study with a known-good baseline so trial 0 has a
    # reasonable score and TPE has a sane starting point. No-op if a
    # completed trial with the same params already exists.
    enqueued_baseline = False
    if not args.no_baseline:
        enqueued_baseline = _enqueue_baseline(study)

    print(f"{'=' * 60}")
    print(f"  Study:              {args.study_name}")
    print(f"  Storage:            {args.storage}")
    print(f"  Trials this run:    {args.trials}")
    print(f"  Episodes / trial:   {args.episodes_per_trial}")
    print(f"  Mode:               {args.training_mode}")
    if args.training_mode == 'single':
        print(f"  Board:              {args.board_size}x{args.board_size}")
    print(f"  Existing trials:    {len(study.trials)}")
    if enqueued_baseline:
        print(
            "  Baseline enqueued:  yes (lr=5e-4, gamma=0.98, "
            "NO_EAT=-0.3, [64,64])"
        )
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
