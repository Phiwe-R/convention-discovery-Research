"""Resumable A4 training matrix: experiments 1/2, seeds 20-22, 100k/500k.

A4 is trained from scratch at EACH budget; old A2 checkpoints NEVER match.
Creates goal-only supervised belief checkpoints automatically if missing.
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('results/goal_only')


def execute(command, log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print('RUN:', ' '.join(map(str, command)), flush=True)
    with log_file.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def completed_checkpoint(experiment, seed, budget, tolerance=256):
    paths = []
    pattern = re.compile(rf'goal_only_seed_{seed}_steps(\d+)\.pt$')
    for path in (ROOT / experiment).glob(f'goal_only_seed_{seed}_steps*.pt'):
        match = pattern.fullmatch(path.name)
        if match and 0 <= int(match.group(1)) - budget <= tolerance:
            paths.append((int(match.group(1)) - budget, path))
    return min(paths)[1] if paths else None


def combine_csv():
    sources = sorted((ROOT / 'training_logs').glob('experiment*/seed*_target*/training_metrics.csv'))
    rows = []
    for path in sources:
        with path.open(newline='', encoding='utf-8') as fp:
            rows.extend(csv.DictReader(fp))
    if rows:
        rows.sort(key=lambda r: (r['experiment'], int(r['target_timesteps']), int(r['seed']), int(r['step'])))
        out = ROOT / 'master_training_curves.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='', encoding='utf-8') as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('Master CSV:', out, 'rows:', len(rows), flush=True)


def ensure_belief(experiment, args):
    suffix = 'exp1' if experiment == 'experiment1' else 'exp2'
    path = ROOT / 'pretrained' / f'goal_only_{suffix}.pt'
    if path.exists():
        print('REUSE goal-only pretrained belief:', path, flush=True)
        return path
    cmd = [sys.executable, '-m', 'training.pretrain_goal_only', '--experiment', experiment,
           '--output', str(path), '--train-per-convention', str(args.train_per_convention),
           '--val-per-convention', str(args.val_per_convention),
           '--test-per-convention', str(args.test_per_convention),
           '--epochs', str(args.pretrain_epochs), '--batch-size', str(args.pretrain_batch_size),
           '--patience', str(args.pretrain_patience), '--device', args.device]
    execute(cmd, ROOT / 'pretrained' / f'{experiment}_pretraining_console.log')
    if not path.is_file():
        raise RuntimeError(f'A4 pretraining produced no checkpoint: {path}')
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments', nargs='+', choices=['experiment1', 'experiment2'], default=['experiment1','experiment2'])
    p.add_argument('--seeds', nargs='+', type=int, default=[20,21,22])
    p.add_argument('--budgets', nargs='+', type=int, default=[100000,500000])
    p.add_argument('--train-per-convention', type=int, default=10000)
    p.add_argument('--val-per-convention', type=int, default=1000)
    p.add_argument('--test-per-convention', type=int, default=1000)
    p.add_argument('--pretrain-epochs', type=int, default=35)
    p.add_argument('--pretrain-patience', type=int, default=6)
    p.add_argument('--pretrain-batch-size', type=int, default=128)
    p.add_argument('--guide-strength', type=float, default=5.)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    trackers = {exp: ensure_belief(exp, args) for exp in args.experiments}
    for experiment in args.experiments:
        for budget in args.budgets:
            for seed in args.seeds:
                old = completed_checkpoint(experiment, seed, budget)
                run_name = f'seed{seed}_target{budget}'
                if old:
                    print('SKIP A3:', experiment, seed, budget, old, flush=True)
                    continue
                command = [sys.executable, '-m', 'training.train_goal_only',
                           '--experiment', experiment, '--timesteps', str(budget),
                           '--seed', str(seed), '--pretrained-belief', str(trackers[experiment]),
                           '--guide-strength', str(args.guide_strength), '--device', args.device]
                execute(command, ROOT / experiment / 'console_logs' / f'{run_name}.log')
                combine_csv()
    combine_csv()
    print('A4 training batch complete. Evaluation is separate.', flush=True)


if __name__ == '__main__':
    main()
