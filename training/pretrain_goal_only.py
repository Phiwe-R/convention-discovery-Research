"""Supervised Agent 4(one with goal belief only) pretraining: goal belief and optional next-action head, NO convention loss.

The model never receives the true convention label and does not predict convention IDs.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# The existing generator balances each convention and each hidden-goal pairing.
# Convention labels are discarded before passing data to our network or losses.
from training.pretrain_balanced_belief import make_dataset, collate


class GoalOnlyPretrainer(nn.Module):
    def __init__(self, obs_dim=5, hidden_dim=64, num_goals=8):
        super().__init__()
        self.belief_gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.goal_head = nn.Linear(hidden_dim, num_goals)
        self.movement_head = nn.Linear(hidden_dim, 5)

    def forward(self, observations):
        hidden, _ = self.belief_gru(observations.float())
        return self.goal_head(hidden), self.movement_head(hidden)


def batch_metrics(model, batch, device):
    observations, lengths, unused_conventions, goals, movements = (x.to(device) for x in batch)
    goal_logits, movement_logits = model(observations)
    times = torch.arange(observations.shape[1], device=device)[None, :]
    valid = times < lengths[:, None]
    expanded_goals = goals[:, None].expand_as(valid)
    last = torch.arange(len(goals), device=device)
    goal_step_ce = F.cross_entropy(goal_logits[valid], expanded_goals[valid])
    goal_final_ce = F.cross_entropy(goal_logits[last, lengths - 1], goals)
    movement_ce = F.cross_entropy(movement_logits[valid], movements[valid])
    # Same final-timestep and auxiliary weighting as Agent 2 (one with both heads), without convention CE.
    loss = goal_step_ce + 1.5 * goal_final_ce + 0.3 * movement_ce
    return {
        'loss': loss, 'goal_step_ce': goal_step_ce, 'goal_final_ce': goal_final_ce,
        'next_action_ce': movement_ce,
        'goal_step_accuracy_pct': 100 * (goal_logits[valid].argmax(-1) == expanded_goals[valid]).float().mean(),
        'goal_final_accuracy_pct': 100 * (goal_logits[last, lengths - 1].argmax(-1) == goals).float().mean(),
        'next_action_accuracy_pct': 100 * (movement_logits[valid].argmax(-1) == movements[valid]).float().mean(),
    }


def validate(model, loader, device):
    model.eval()
    sums = {}
    total = 0
    with torch.no_grad():
        for batch in loader:
            metrics = batch_metrics(model, batch, device)
            weight = len(batch[1])
            total += weight
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.) + float(v) * weight
    return {k: v / total for k, v in sums.items()}


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timestep_accuracy(model, loader, device):
    counts = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            observations, lengths, _, goals, _ = (x.to(device) for x in batch)
            logits, _ = model(observations)
            for t in range(observations.shape[1]):
                mask = lengths > t
                if mask.any():
                    hit = int((logits[mask, t].argmax(-1) == goals[mask]).sum().item())
                    count = int(mask.sum().item())
                    old = counts.setdefault(t + 1, [0, 0])
                    old[0] += count
                    old[1] += hit
    return [{'timestep': t, 'observations': n, 'goal_id_accuracy_pct': 100 * hits / n}
            for t, (n, hits) in sorted(counts.items())]


def pretrain(experiment, output, train_per_convention=10000,
             val_per_convention=1000, test_per_convention=1000, epochs=35,
             batch_size=128, patience=6, seed=20, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cpu':
        torch.set_num_threads(1)
    dev = torch.device(device)
    n = {'experiment1': 3, 'experiment2': 6}[experiment]
    print('A4: generating balanced training/validation/test episodes', flush=True)
    training = make_dataset(experiment, train_per_convention, 200000, seed)
    validation = make_dataset(experiment, val_per_convention, 400000, seed)
    testing = make_dataset(experiment, test_per_convention, 800000, seed)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, collate_fn=collate)
    train_eval_loader = DataLoader(training, batch_size=batch_size, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(testing, batch_size=batch_size, collate_fn=collate)
    model = GoalOnlyPretrainer().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(Path('runs/goal_only/pretrain') / experiment))
    history = []
    best_loss = float('inf')
    best_state = None
    best_epoch = 0
    stale = 0
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                metrics = batch_metrics(model, batch, dev)
                optimizer.zero_grad(set_to_none=True)
                metrics['loss'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            train_stats = validate(model, train_eval_loader, dev)
            val_stats = validate(model, val_loader, dev)
            row = {'epoch': epoch}
            for split, stats in [('train', train_stats), ('val', val_stats)]:
                for name, value in stats.items():
                    row[f'{split}_{name}'] = value
                    tb.add_scalar(f'{split}/{name}', value, epoch)
            history.append(row)
            tb.flush()
            print(f'Epoch {epoch:02d} | val goal-step {val_stats["goal_step_accuracy_pct"]:.1f}% '
                  f'| val goal-final {val_stats["goal_final_accuracy_pct"]:.1f}% '
                  f'| val CE {val_stats["goal_step_ce"]:.3f} | next-action CE {val_stats["next_action_ce"]:.3f}', flush=True)
            if val_stats['loss'] < best_loss - 0.001:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_loss = val_stats['loss']
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print('Early stopping', flush=True)
                    break
        if best_state is None:
            raise RuntimeError('No pretraining epochs completed')
        model.load_state_dict(best_state)
        test = validate(model, test_loader, dev)
        step_curve = timestep_accuracy(model, test_loader, dev)
        write_csv(path.with_name(path.stem + '_history.csv'), history)
        write_csv(path.with_name(path.stem + '_test_by_timestep.csv'), step_curve)
        for row in step_curve:
            tb.add_scalar('test/goal_by_timestep', row['goal_id_accuracy_pct'], row['timestep'])
        torch.save({
            'architecture': 'goal_only_belief_pretrain_v1', 'experiment': experiment,
            'num_conventions_environment': n, 'num_goals': 8,
            'belief_gru': model.belief_gru.state_dict(),
            'goal_head': model.goal_head.state_dict(),
            'movement_head': model.movement_head.state_dict(),
            'best_epoch': best_epoch, 'test_metrics': test,
            'convention_head_present': False,
        }, path)
        print(f'Saved A4 goal-only beliefs: {path} | test={test}', flush=True)
        return path
    finally:
        tb.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', choices=['experiment1', 'experiment2'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-per-convention', type=int, default=10000)
    p.add_argument('--val-per-convention', type=int, default=1000)
    p.add_argument('--test-per-convention', type=int, default=1000)
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--device', default='cpu')
    pretrain(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
