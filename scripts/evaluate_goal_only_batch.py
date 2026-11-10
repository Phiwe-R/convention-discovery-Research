"""Evaluate available A4 checkpoints on matched held-out seeds; preserve per-episode CSVs."""
import argparse
import csv
from pathlib import Path

from scripts.train_goal_only_batch import completed_checkpoint
from evaluation.evaluate_goal_only import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments', nargs='+', choices=['experiment1','experiment2'], default=['experiment1','experiment2'])
    p.add_argument('--seeds', nargs='+', type=int, default=[20,21,22])
    p.add_argument('--budgets', nargs='+', type=int, default=[100000,500000])
    p.add_argument('--modes', nargs='+', choices=['real','uniform'], default=['real','uniform'])
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--force', action='store_true', help='Re-evaluate even if summary.json exists')
    args = p.parse_args()
    root = Path('results/goal_only/evaluation')
    rows = []
    for experiment in args.experiments:
        for budget in args.budgets:
            for seed in args.seeds:
                model = completed_checkpoint(experiment, seed, budget)
                if model is None:
                    print(f'MISSING A3 checkpoint: {experiment} seed={seed} budget={budget}; skip', flush=True)
                    continue
                for mode in args.modes:
                    folder = root / experiment / f'seed{seed}_target{budget}' / mode
                    summary_path = folder / 'summary.json'
                    if summary_path.exists() and not args.force:
                        import json
                        old = json.loads(summary_path.read_text(encoding='utf-8'))
                        if (old.get('episodes') == args.episodes and old.get('evaluation_seed_start') == args.seed_start
                                and old.get('checkpoint') == str(model)):
                            print('SKIP evaluated:', summary_path, flush=True)
                            rows.append(old)
                            continue
                    result = evaluate(model, episodes=args.episodes, seed_start=args.seed_start,
                                      belief_mode=mode, device=args.device, output_dir=folder)
                    rows.append(result)
                    print(f'{experiment} seed={seed} {budget} mode={mode} success={result["success_pct"]:.2f}% '
                          f'goalID={result["final_goal_id_pct"]:.2f}%', flush=True)
                    save_master(root, rows)
    save_master(root, rows)
    print('A3 evaluation batch complete.', flush=True)


def save_master(root, rows):
    if not rows:
        return
    dest = root / 'master_evaluation_results.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r['experiment'], r['target_timesteps'], r['seed'], r['belief_mode']))
    with dest.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
