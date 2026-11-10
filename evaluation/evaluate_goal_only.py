"""
this is an old code before we ran run_a3_evaluation.bat  so ignore it.

Evaluate Agent 4  goal-only guided policy; real vs uniform GOAL beliefs only.

Outputs one episode-level CSV, one timestep identification CSV, and summary JSON.
No true convention or true learner-goal label is supplied to the actor.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from agents.goal_only_guided_actor import GoalOnlyGuidedActor
from configs.experiments import make_env


def evaluate(checkpoint, episodes=1000, seed_start=10000, belief_mode='real',
             device='cpu', output_dir=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get('architecture') != 'goal_only_guided_clean_v1':
        raise ValueError('Not an A3 goal-only-guided checkpoint')
    experiment = saved['experiment']
    actor = GoalOnlyGuidedActor(guide_strength=float(saved['guide_strength'])).to(device)
    actor.load_state_dict(saved['actor'], strict=True)
    actor.eval()
    environment = make_env(experiment)
    rows = []
    timestep = defaultdict(lambda: [0, 0, 0.])
    try:
        with torch.no_grad():
            for seed in range(seed_start, seed_start + episodes):
                obs, info = environment.reset(seed=seed)
                goal_id = int(info['learner_goal_id'])  # evaluator-only scoring label
                memory = actor.initial_memory(1, torch.device(device))
                done = truncated = False
                reward_total = 0.
                hit = 0
                steps = 0
                hit_steps = 0
                final_true_probability = 0.
                while not (done or truncated):
                    inp = torch.as_tensor(np.asarray(obs['learner'], dtype=np.float32), device=device).unsqueeze(0)
                    action, _, _, memory = actor.get_action(
                        inp, memory, deterministic=True, belief_mode=belief_mode
                    )
                    probabilities = memory[0][0]
                    hit = int(probabilities.argmax(-1).item() == goal_id)
                    correct_prob = float(probabilities[goal_id])
                    hit_steps += hit
                    steps += 1
                    bucket = timestep[steps]
                    bucket[0] += 1
                    bucket[1] += hit
                    bucket[2] += correct_prob
                    final_true_probability = correct_prob
                    obs, reward, done, truncated, _ = environment.step(int(action.item()))
                    reward_total += float(reward)
                rows.append({
                    'evaluation_seed': seed, 'success': int(done), 'reward': reward_total,
                    'length': steps, 'final_goal_id_correct': hit,
                    'step_goal_id_correct_count': hit_steps,
                    'final_true_goal_probability': final_true_probability,
                })
    finally:
        environment.close()
    total_steps = sum(r['length'] for r in rows)
    output = {
        'architecture': saved['architecture'], 'experiment': experiment,
        'seed': saved['seed'], 'target_timesteps': saved.get('target_timesteps', saved['timesteps']),
        'checkpoint': str(checkpoint), 'belief_mode': belief_mode,
        'episodes': episodes, 'evaluation_seed_start': seed_start,
        'success_pct': 100 * np.mean([r['success'] for r in rows]),
        'mean_reward': float(np.mean([r['reward'] for r in rows])),
        'mean_length': float(np.mean([r['length'] for r in rows])),
        'final_goal_id_pct': 100 * np.mean([r['final_goal_id_correct'] for r in rows]),
        'step_goal_id_pct': 100 * sum(r['step_goal_id_correct_count'] for r in rows) / total_steps,
        'mean_true_goal_probability_final': float(np.mean([r['final_true_goal_probability'] for r in rows])),
        'convention_head_present': False,
    }
    if output_dir:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'episodes.csv').open('w', newline='', encoding='utf-8') as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        step_rows = [{
            'timestep': t, 'observations': n,
            'goal_id_accuracy_pct': 100 * correct / n,
            'mean_true_goal_probability': prob / n,
        } for t, (n, correct, prob) in sorted(timestep.items())]
        with (folder / 'goal_identification_by_timestep.csv').open('w', newline='', encoding='utf-8') as fp:
            w = csv.DictWriter(fp, fieldnames=list(step_rows[0]))
            w.writeheader()
            w.writerows(step_rows)
        (folder / 'summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--belief-mode', choices=['real', 'uniform', 'uniform_goal'], default='real')
    p.add_argument('--output-dir')
    p.add_argument('--device', default='cpu')
    result = evaluate(**vars(p.parse_args()))
    for key, value in result.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
