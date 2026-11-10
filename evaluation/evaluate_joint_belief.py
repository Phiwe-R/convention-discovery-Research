"""
ignore this
Evaluate Agent 2: joint convention and learner-goal belief predictor on held-out episodes.

python -m evaluation.evaluate_joint_belief --checkpoint results/joint_belief/experiment2/joint_seed_20_steps2400.pt --episodes 100
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch

from configs.experiments import make_env
from agents.joint_belief_agent import JointBeliefActor


def evaluate(checkpoint, episodes=100, seed_start=10000, mode='real', device='cpu', output=None):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get('architecture') != 'joint_belief_v1':
        raise ValueError('This evaluation requires a joint_belief_v1 checkpoint')
    experiment = saved['experiment']
    actor = JointBeliefActor(num_conventions=saved['num_conventions']).to(device)
    actor.load_state_dict(saved['actor'])
    actor.eval()
    env = make_env(experiment)
    records = []
    try:
        with torch.no_grad():
            for seed in range(seed_start, seed_start + episodes):
                obs, info = env.reset(seed=seed)
                true_conv = int(obs['convention'][0])
                true_goal = int(info['learner_goal_id'])
                memory = actor.initial_memory(1, device)
                conv_correct, goal_correct = [], []
                true_conv_probs, true_goal_probs = [], []
                total_reward = 0.0
                done = truncated = False
                while not (done or truncated):
                    local = torch.as_tensor(np.asarray(obs['learner'], dtype=np.float32), device=device).unsqueeze(0)
                    action, _, _, memory = actor.get_action(local, memory, deterministic=True, belief_mode=mode)
                    c, g = memory[:2]
                    conv_correct.append(int(c.argmax().item() == true_conv))
                    goal_correct.append(int(g.argmax().item() == true_goal))
                    true_conv_probs.append(float(c[0, true_conv].item()))
                    true_goal_probs.append(float(g[0, true_goal].item()))
                    obs, reward, done, truncated, _ = env.step(int(action.item()))
                    total_reward += float(reward)
                records.append({
                    'seed': seed, 'success': int(done), 'reward': total_reward,
                    'length': len(conv_correct), 'convention_correct_final': conv_correct[-1],
                    'goal_correct_final': goal_correct[-1],
                    'convention_correct_all': float(np.mean(conv_correct)),
                    'goal_correct_all': float(np.mean(goal_correct)),
                    'true_convention_prob_final': true_conv_probs[-1],
                    'true_goal_prob_final': true_goal_probs[-1],
                })
    finally:
        env.close()
    metric = lambda field: float(np.mean([r[field] for r in records]))
    summary = {'checkpoint': str(checkpoint), 'experiment': experiment, 'mode': mode,
               'episodes': episodes, 'success_pct': 100 * metric('success'),
               'mean_reward': metric('reward'),
               'final_convention_id_pct': 100 * metric('convention_correct_final'),
               'final_goal_id_pct': 100 * metric('goal_correct_final'),
               'step_convention_id_pct': 100 * np.average(
                   [r['convention_correct_all'] for r in records],
                   weights=[r['length'] for r in records]),
               'step_goal_id_pct': 100 * np.average(
                   [r['goal_correct_all'] for r in records],
                   weights=[r['length'] for r in records]),
               'true_convention_probability': metric('true_convention_prob_final'),
               'true_goal_probability': metric('true_goal_prob_final')}
    if output:
        outpath = Path(output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with outpath.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--belief-mode', choices=['real','uniform','uniform_goal','uniform_convention'], default='real')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default=None)
    a = p.parse_args()
    for k,v in evaluate(a.checkpoint, a.episodes, a.seed_start, a.belief_mode, a.device, a.output).items():
        print(f'{k}: {v}')

if __name__ == '__main__': main()
