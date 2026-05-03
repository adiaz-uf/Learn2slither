"""
Inspect a completed (or in-progress) Optuna study.

Prints:
  - Summary (num trials, completed, pruned, failed).
  - Best trial: rolling-200 avg + parameters + ready-to-paste training cmd.
  - Top-10 trials sorted by value.
  - Parameter importance (which hyperparameters actually moved the metric).
  - Per-parameter "good zone": median and IQR among the top-25% of trials.

The "good zone" tells you whether the best params are an outlier or whether
the surrounding region of search space is consistently strong — useful for
picking robust values to keep when you train the real model.

Usage:
    python src/AI_Model/inspect_study.py
    python src/AI_Model/inspect_study.py --study-name snake-multi
    python src/AI_Model/inspect_study.py \\
        --storage sqlite:///optuna_studies/snake-single-10.db
"""

import argparse
import os
import sys
import statistics

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import optuna  # noqa: E402


def _format_param(value):
    if isinstance(value, float):
        if abs(value) < 1e-2 or abs(value) >= 1e3:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def _print_summary(study):
    states = {s.name: 0 for s in optuna.trial.TrialState}
    for t in study.trials:
        states[t.state.name] += 1

    print(f"\n{'=' * 60}")
    print(f"Study: {study.study_name}")
    print(f"  Total trials: {len(study.trials)}")
    for name, count in states.items():
        if count > 0:
            print(f"    {name.lower():12s} {count}")
    print(f"{'=' * 60}\n")


def _print_best(study):
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        print("No completed trial yet.\n")
        return None

    print("BEST TRIAL")
    print("-" * 60)
    print(f"  Value (rolling-200 avg): {study.best_value:.3f}")
    high = study.best_trial.user_attrs.get('high_score')
    if high is not None:
        print(f"  Highest score in trial:  {high}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {_format_param(v)}")
    print()
    return study.best_params


def _print_top_n(study, n=10):
    completed = sorted(
        [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ],
        key=lambda t: t.value,
        reverse=True,
    )
    if not completed:
        return

    print(f"TOP {min(n, len(completed))} TRIALS")
    print("-" * 60)
    header = (
        f"  {'rank':>4}  {'value':>7}  "
        f"{'lr':>9}  {'gamma':>6}  "
        f"{'batch':>5}  {'no_eat':>7}  "
        f"{'layers':<25}"
    )
    print(header)
    for i, t in enumerate(completed[:n]):
        p = t.params
        layers = [
            p.get(f'units_l{j}')
            for j in range(p.get('num_layers', 0))
        ]
        layer_str = ','.join(str(u) for u in layers if u is not None)
        line = (
            f"  {i + 1:>4}  {t.value:>7.3f}  "
            f"{p.get('learning_rate', 0):>9.5f}  "
            f"{p.get('gamma', 0):>6.3f}  "
            f"{p.get('batch_size', 0):>5}  "
            f"{p.get('no_eat', 0):>7.2f}  "
            f"{layer_str:<25}"
        )
        print(line)
    print()


def _print_importance(study):
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed) < 5:
        print("Need at least 5 completed trials for importance.\n")
        return
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:
        print(f"Could not compute importance: {exc}\n")
        return

    print("PARAMETER IMPORTANCE")
    print("(higher = this hyperparameter moved the metric the most)")
    print("-" * 60)
    for k, v in importances.items():
        bar = '#' * int(v * 40)
        print(f"  {k:20s} {v:.3f}  {bar}")
    print()


def _print_good_zone(study, top_pct=0.25):
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed) < 8:
        print("Need at least 8 completed trials for good-zone analysis.\n")
        return

    sorted_completed = sorted(
        completed, key=lambda t: t.value, reverse=True
    )
    cutoff = max(2, int(len(sorted_completed) * top_pct))
    top = sorted_completed[:cutoff]

    print(f"GOOD ZONE — median & range across top-{cutoff} trials")
    print("(if the best params lie inside these ranges they are robust;")
    print(" if they are outliers consider picking the median instead)")
    print("-" * 60)

    numeric_keys = ['learning_rate', 'gamma', 'no_eat']
    for k in numeric_keys:
        vals = [t.params.get(k) for t in top if k in t.params]
        if not vals:
            continue
        med = statistics.median(vals)
        lo, hi = min(vals), max(vals)
        print(f"  {k:20s} median={_format_param(med)}  "
              f"range=[{_format_param(lo)}, {_format_param(hi)}]")

    # Categorical: batch_size — show counts
    batch_counts = {}
    for t in top:
        b = t.params.get('batch_size')
        if b is not None:
            batch_counts[b] = batch_counts.get(b, 0) + 1
    if batch_counts:
        most_freq = max(batch_counts.items(), key=lambda x: x[1])
        counts_str = ", ".join(
            f"{k}: {v}"
            for k, v in sorted(batch_counts.items())
        )
        print(f"  {'batch_size':20s} mode={most_freq[0]}  "
              f"counts: {counts_str}")

    # Architecture summary
    layer_counts = {}
    for t in top:
        n = t.params.get('num_layers')
        if n is not None:
            layer_counts[n] = layer_counts.get(n, 0) + 1
    if layer_counts:
        counts_str = ", ".join(
            f"{k}-layer: {v}"
            for k, v in sorted(layer_counts.items())
        )
        print(f"  {'num_layers':20s} {counts_str}")
    print()


def _print_train_command(study, args):
    if not study.best_params:
        return
    bp = study.best_params
    hidden_units = [
        bp[f'units_l{i}']
        for i in range(bp['num_layers'])
    ]
    hidden_arg = ','.join(str(u) for u in hidden_units)
    cmd = [
        "python src/AI_Model/main.py",
        "--sessions 10000",
    ]
    if args.training_mode == 'multi':
        cmd.append("--training-mode multi")
    else:
        cmd.append("--training-mode single")
        cmd.append(f"--board-size {args.board_size}")
    cmd.extend([
        f"--learning-rate {bp['learning_rate']:.6f}",
        f"--gamma {bp['gamma']:.4f}",
        f"--batch-size {bp['batch_size']}",
        f"--no-eat-reward {bp['no_eat']:.3f}",
        f"--hidden-layers {hidden_arg}",
    ])
    print("READY-TO-PASTE TRAINING COMMAND")
    print("-" * 60)
    print("    " + " \\\n        ".join(cmd))
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Inspect an Optuna tuning study.'
    )
    parser.add_argument(
        '--study-name', type=str, default=None,
        help=(
            'Study name (default: snake-single-<board> or snake-multi).'
        ),
    )
    parser.add_argument(
        '--training-mode', choices=['single', 'multi'], default='single',
    )
    parser.add_argument(
        '--board-size', type=int, default=10, choices=[10, 14, 18],
    )
    parser.add_argument(
        '--storage', type=str, default=None,
        help='SQLite storage URL.',
    )
    parser.add_argument(
        '--top', type=int, default=10,
        help='Number of top trials to print (default: 10).',
    )
    args = parser.parse_args()

    if args.study_name is None:
        if args.training_mode == 'multi':
            args.study_name = 'snake-multi'
        else:
            args.study_name = f'snake-single-{args.board_size}'

    if args.storage is None:
        args.storage = (
            f'sqlite:///optuna_studies/{args.study_name}.db'
        )

    if not os.path.exists(args.storage.replace('sqlite:///', '')):
        print(f"No study DB found at {args.storage}")
        print("Run `make tune` first.")
        sys.exit(1)

    study = optuna.load_study(
        study_name=args.study_name, storage=args.storage
    )

    _print_summary(study)
    _print_best(study)
    _print_top_n(study, n=args.top)
    _print_importance(study)
    _print_good_zone(study)
    _print_train_command(study, args)


if __name__ == '__main__':
    main()
