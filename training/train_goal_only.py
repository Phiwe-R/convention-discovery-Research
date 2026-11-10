"""Train A4 goal-only belief-guided PPO (NO convention head).

Uses A4's environment, critic, PPO coefficients, goal-distance guide, and
training budgets. CSV + TensorBoard are written after every PPO update.
"""
import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from agents.centralized_critic import CentralizedCritic, build_critic_state
from agents.goal_only_guided_actor import GoalOnlyGuidedActor
from configs.experiments import make_env
from training.train_mappo import discounted_gae


FIELDS = [
    'architecture', 'experiment', 'seed', 'target_timesteps', 'step', 'update', 'episodes',
    'cumulative_avg_reward', 'recent_avg_reward',
    'cumulative_goal_success_pct', 'recent_goal_success_pct',
    'cumulative_goal_id_step_pct', 'recent_goal_id_step_pct',
    'cumulative_goal_id_episode_pct', 'recent_goal_id_episode_pct',
    'policy_loss', 'value_loss', 'policy_entropy', 'entropy_loss',
    'belief_goal_ce', 'ppo_total_loss', 'supervised_action_cross_entropy_loss', 'learning_rate',
]


def pct(n, d):
    return 100. * n / d if d else 0.


def collect_episode(env, actor, critic, n_conventions, device, seed):
    obs, info = env.reset(seed=seed)
    goal_id = int(info['learner_goal_id'])
    memory = actor.initial_memory(1, device)
    observations, states, actions, logps, rewards = [], [], [], [], []
    goal_hits = []
    terminated = truncated = False
    with torch.no_grad():
        while not (terminated or truncated):
            states.append(build_critic_state(obs, len(rewards), env.env.max_steps, n_conventions))
            local = np.asarray(obs['learner'], dtype=np.float32)
            x = torch.as_tensor(local, device=device).unsqueeze(0)
            action, logp, _, memory = actor.get_action(x, memory)
            actions.append(int(action.item()))
            logps.append(float(logp.item()))
            observations.append(local)
            goal_hits.append(int(memory[0].argmax(-1).item() == goal_id))
            obs, reward, terminated, truncated, _ = env.step(actions[-1])
            rewards.append(float(reward))
        final_state = build_critic_state(obs, len(rewards), env.env.max_steps, n_conventions)
        state_tensor = torch.as_tensor(np.asarray(states), device=device, dtype=torch.float32)
        values = critic(state_tensor)
        next_value = critic(torch.as_tensor(final_state, device=device, dtype=torch.float32))
        rewards_tensor = torch.as_tensor(rewards, device=device, dtype=torch.float32)
        advantages, returns = discounted_gae(rewards_tensor, values, next_value, terminated)
    return {
        'observations': torch.as_tensor(np.asarray(observations), device=device, dtype=torch.float32),
        'states': state_tensor, 'actions': torch.as_tensor(actions, device=device, dtype=torch.long),
        'old_logps': torch.as_tensor(logps, device=device, dtype=torch.float32),
        'advantages': advantages, 'returns': returns,
        'goal_id': goal_id, 'reward': sum(rewards), 'success': bool(terminated),
        'length': len(rewards), 'goal_correct_steps': goal_hits, 'goal_correct_final': goal_hits[-1],
    }


def update_ppo(actor, critic, optimizer, episodes, epochs=4):
    advantages = torch.cat([ep['advantages'] for ep in episodes])
    mean = advantages.mean()
    scale = advantages.std(unbiased=False).clamp_min(1e-8)
    step_count = sum(ep['length'] for ep in episodes)
    reports = []
    actor.train()
    for _ in range(epochs):
        policy = torch.zeros((), device=advantages.device)
        value = torch.zeros((), device=advantages.device)
        entropy = torch.zeros((), device=advantages.device)
        for ep in episodes:
            obs = ep['observations'].unsqueeze(0)
            logits, _, _ = actor(obs)
            new_lp = F.log_softmax(logits, dim=-1).gather(
                dim=-1, index=ep['actions'][None, :, None]
            ).squeeze(0).squeeze(-1)
            ratios = (new_lp - ep['old_logps']).exp()
            adv = (ep['advantages'] - mean) / scale
            policy = policy - torch.minimum(ratios * adv, ratios.clamp(0.8, 1.2) * adv).sum() / step_count
            value = value + (critic(ep['states']) - ep['returns']).square().sum() / (2 * step_count)
            probs = logits.softmax(-1)
            entropy = entropy - (probs * F.log_softmax(logits, -1)).sum() / step_count
        entropy_term = -0.01 * entropy
        total = policy + 0.5 * value + entropy_term
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        clip_grad_norm_(actor.actor_parameters() + list(critic.parameters()), 0.5)
        optimizer.step()
        reports.append([float(t.detach()) for t in [policy, value, entropy, entropy_term, total]])

    # Diagnostic ONLY. Goal labels do not feed gradients into frozen belief or policy.
    actor.eval()
    goal_ce_sum = 0.
    with torch.no_grad():
        for ep in episodes:
            _, logits, _ = actor(ep['observations'].unsqueeze(0))
            labels = torch.full((ep['length'],), ep['goal_id'], dtype=torch.long, device=advantages.device)
            goal_ce_sum += float(F.cross_entropy(logits.squeeze(0), labels, reduction='sum'))
    report = np.asarray(reports).mean(0)
    return {
        'policy_loss': float(report[0]), 'value_loss': float(report[1]),
        'policy_entropy': float(report[2]), 'entropy_loss': float(report[3]),
        'ppo_total_loss': float(report[4]), 'belief_goal_ce': goal_ce_sum / step_count,
        'supervised_action_cross_entropy_loss': 0.,
    }


def summarize(episodes):
    n = len(episodes)
    return {
        'avg_reward': sum(e['reward'] for e in episodes) / n,
        'goal_success_pct': pct(sum(e['success'] for e in episodes), n),
        'goal_id_step_pct': pct(sum(sum(e['goal_correct_steps']) for e in episodes),
                                sum(e['length'] for e in episodes)),
        'goal_id_episode_pct': pct(sum(e['goal_correct_final'] for e in episodes), n),
    }


def train(experiment, timesteps, seed, pretrained_belief, guide_strength=5.0,
          learning_rate=2.5e-4, episodes_per_update=8, update_epochs=4,
          recent_window=100, print_interval=20, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cpu':
        torch.set_num_threads(1)
    device = torch.device(device)
    n = {'experiment1': 3, 'experiment2': 6}[experiment]
    belief = torch.load(pretrained_belief, map_location=device, weights_only=True)
    if belief.get('architecture') != 'goal_only_belief_pretrain_v1' or belief.get('experiment') != experiment:
        raise ValueError('A3 requires a goal-only belief checkpoint for the matching experiment')
    actor = GoalOnlyGuidedActor(guide_strength=guide_strength).to(device)
    actor.belief_gru.load_state_dict(belief['belief_gru'])
    actor.goal_head.load_state_dict(belief['goal_head'])
    for param in actor.belief_parameters():
        param.requires_grad_(False)
    critic = CentralizedCritic(num_conventions=n).to(device)
    optimizer = torch.optim.Adam(actor.actor_parameters() + list(critic.parameters()), lr=learning_rate, eps=1e-5)
    run_id = f'seed{seed}_target{timesteps}'
    csv_dir = Path('results/goal_only/training_logs') / experiment / run_id
    tb_dir = Path('runs/goal_only/ppo') / experiment / run_id
    csv_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / 'training_metrics.csv'
    (csv_dir / 'run_config.json').write_text(json.dumps({
        'architecture': 'goal_only_guided_clean_v1', 'experiment': experiment,
        'seed': seed, 'target_timesteps': timesteps, 'guide_strength': guide_strength,
        'pretrained_belief': str(pretrained_belief), 'belief_frozen': True,
        'convention_head_present': False, 'imitation_enabled': False,
        'critic_training_only_convention': True,
    }, indent=2), encoding='utf-8')
    writer = SummaryWriter(str(tb_dir))
    env = make_env(experiment)
    count = steps = updates = hits = goal_step_correct = goal_final_correct = 0
    reward_total = 0.
    recent = deque(maxlen=recent_window)
    try:
        with csv_path.open('w', newline='', encoding='utf-8') as stream:
            log = csv.DictWriter(stream, fieldnames=FIELDS)
            log.writeheader()
            while steps < timesteps:
                episodes = []
                for _ in range(episodes_per_update):
                    if steps >= timesteps:
                        break
                    ep = collect_episode(env, actor, critic, n, device, seed + count)
                    episodes.append(ep)
                    recent.append(ep)
                    steps += ep['length']
                    count += 1
                    hits += int(ep['success'])
                    reward_total += ep['reward']
                    goal_step_correct += sum(ep['goal_correct_steps'])
                    goal_final_correct += ep['goal_correct_final']
                losses = update_ppo(actor, critic, optimizer, episodes, epochs=update_epochs)
                updates += 1
                row = {
                    'architecture': 'goal_only_guided_clean_v1',
                    'experiment': experiment, 'seed': seed, 'target_timesteps': timesteps,
                    'step': steps, 'update': updates, 'episodes': count,
                    'cumulative_avg_reward': reward_total / count,
                    'cumulative_goal_success_pct': pct(hits, count),
                    'cumulative_goal_id_step_pct': pct(goal_step_correct, steps),
                    'cumulative_goal_id_episode_pct': pct(goal_final_correct, count),
                    **{f'recent_{k}': v for k, v in summarize(recent).items()},
                    **losses, 'learning_rate': optimizer.param_groups[0]['lr'],
                }
                log.writerow(row)
                stream.flush()
                for key, val in row.items():
                    if key not in ('architecture', 'experiment', 'seed', 'target_timesteps', 'step', 'update'):
                        writer.add_scalar(key, val, steps)
                writer.flush()
                if updates % print_interval == 0 or steps >= timesteps:
                    print(f'A3 {experiment} seed={seed} Steps {steps:>7} | Episodes {count:>6} '
                          f'| Success {row["cumulative_goal_success_pct"]:.1f}% '
                          f'| Recent {row["recent_goal_success_pct"]:.1f}% '
                          f'| Goal ID step {row["recent_goal_id_step_pct"]:.1f}% '
                          f'| Goal ID final {row["recent_goal_id_episode_pct"]:.1f}% '
                          f'| Reward {row["recent_avg_reward"]:.3f} '
                          f'| Policy {row["policy_loss"]:.3f} '
                          f'| Goal CE {row["belief_goal_ce"]:.3f}', flush=True)
    finally:
        env.close()
        writer.close()
    checkpoint = Path('results/goal_only') / experiment / f'goal_only_seed_{seed}_steps{steps}.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'architecture': 'goal_only_guided_clean_v1', 'experiment': experiment,
        'actor': actor.state_dict(), 'critic': critic.state_dict(),
        'optimizer': optimizer.state_dict(), 'seed': seed,
        'timesteps': steps, 'target_timesteps': timesteps, 'num_conventions_environment': n,
        'num_goals': 8, 'guide_strength': guide_strength,
        'pretrained_belief': str(pretrained_belief), 'imitation_loss_enabled': False,
        'convention_head_present': False, 'training_csv': str(csv_path),
        'tensorboard_dir': str(tb_dir),
    }, checkpoint)
    print(f'Saved A3 checkpoint: {checkpoint}\nCSV: {csv_path}\nTensorBoard: {tb_dir}', flush=True)
    return checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', choices=['experiment1', 'experiment2'], required=True)
    p.add_argument('--timesteps', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--pretrained-belief', required=True)
    p.add_argument('--guide-strength', type=float, default=5.0)
    p.add_argument('--learning-rate', type=float, default=2.5e-4)
    p.add_argument('--episodes-per-update', type=int, default=8)
    p.add_argument('--update-epochs', type=int, default=4)
    p.add_argument('--recent-window', type=int, default=100)
    p.add_argument('--print-interval', type=int, default=20)
    p.add_argument('--device', default='cpu')
    train(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
