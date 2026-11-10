"""
Leave this code alone, its one of the wrong ones when i was experimenting in earlier stages.

Train joint convention + hidden-goal beliefs with a detached-belief PPO actor.
and it doesnt have goal guidance function
Example: python -m training.train_joint_belief --experiment experiment2 --timesteps 2400 --seed 20
This creates new checkpoints; never overwrites the original implicit/explicit baselines.
"""
import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agents.centralized_critic import CentralizedCritic, build_critic_state
from agents.joint_belief_agent import JointBeliefActor
from configs.experiments import make_env
from training.train_mappo import discounted_gae


def build_models(experiment, device):
    n = {'experiment1': 3, 'experiment2': 6}[experiment]
    return JointBeliefActor(num_conventions=n).to(device), CentralizedCritic(num_conventions=n).to(device)


def collect_episode(env, actor, critic, n, device, seed):
    obs, info = env.reset(seed=seed)
    convention = int(obs['convention'][0])
    # This is the hidden learner-goal CLASS (0..7), not its coordinates.
    goal_id = int(info['learner_goal_id'])
    memory = actor.initial_memory(1, device)
    observations, states, actions, logps, rewards = [], [], [], [], []
    id_hits, goal_hits = [], []
    terminated = truncated = False
    with torch.no_grad():
        while not (terminated or truncated):
            states.append(build_critic_state(obs, len(rewards), env.env.max_steps, n))
            local = np.asarray(obs['learner'], dtype=np.float32)
            local_tensor = torch.tensor(local, device=device).unsqueeze(0)
            action, logp, _, memory = actor.get_action(local_tensor, memory)
            actions.append(int(action.item()))
            logps.append(float(logp.item()))
            observations.append(local)
            id_hits.append(int(memory[0].argmax(-1).item() == convention))
            goal_hits.append(int(memory[1].argmax(-1).item() == goal_id))
            obs, reward, terminated, truncated, _ = env.step(actions[-1])
            rewards.append(float(reward))
        final_state = build_critic_state(obs, len(rewards), env.env.max_steps, n)
        states = torch.as_tensor(np.asarray(states), device=device, dtype=torch.float32)
        values = critic(states)
        next_value = critic(torch.as_tensor(final_state, device=device, dtype=torch.float32))
        rewards_tensor = torch.as_tensor(rewards, device=device, dtype=torch.float32)
        advantages, returns = discounted_gae(rewards_tensor, values, next_value, terminated)
    return {
        'observations': torch.tensor(np.asarray(observations), device=device),
        'states': states,
        'actions': torch.tensor(actions, device=device, dtype=torch.long),
        'old_logps': torch.tensor(logps, device=device),
        'advantages': advantages, 'returns': returns,
        'convention': convention, 'goal_id': goal_id,
        'success': bool(terminated), 'reward': sum(rewards), 'length': len(rewards),
        'id_correct_final': id_hits[-1], 'goal_correct_final': goal_hits[-1],
        'id_correct_steps': id_hits, 'goal_correct_steps': goal_hits,
    }


def update_models(actor, critic, policy_optimizer, belief_optimizer, episodes, *,
                  epochs=4, clip_coef=0.2, value_coef=0.5, entropy_coef=0.01,
                  convention_coef=1.0, goal_coef=1.0, max_grad_norm=0.5):
    all_adv = torch.cat([e['advantages'] for e in episodes])
    mean = all_adv.mean()
    std = all_adv.std(unbiased=False).clamp_min(1e-8)
    total_steps = sum(e['length'] for e in episodes)
    for _ in range(epochs):
        policy_loss = value_loss = entropy = conv_ce = goal_ce = 0.0
        for ep in episodes:
            obs = ep['observations'].unsqueeze(0)
            actions = ep['actions'].unsqueeze(0)
            new_logp, ent, conv_logits, goal_logits = actor.evaluate_actions(obs, actions)
            new_logp = new_logp.squeeze(0)
            ratios = (new_logp - ep['old_logps']).exp()
            adv = (ep['advantages'] - mean) / std
            policy_loss = policy_loss - torch.minimum(
                ratios * adv, ratios.clamp(1 - clip_coef, 1 + clip_coef) * adv
            ).sum() / total_steps
            value_loss = value_loss + (critic(ep['states']) - ep['returns']).square().sum() / (2 * total_steps)
            entropy = entropy + ent.sum() / total_steps
            target_c = torch.full((ep['length'],), ep['convention'], device=obs.device, dtype=torch.long)
            target_g = torch.full((ep['length'],), ep['goal_id'], device=obs.device, dtype=torch.long)
            conv_ce = conv_ce + F.cross_entropy(conv_logits.squeeze(0), target_c, reduction='sum') / total_steps
            goal_ce = goal_ce + F.cross_entropy(goal_logits.squeeze(0), target_g, reduction='sum') / total_steps

        # Separate graphs: actor sees beliefs only through detach().
        ppo = policy_loss + value_coef * value_loss - entropy_coef * entropy
        policy_optimizer.zero_grad()
        ppo.backward()
        nn.utils.clip_grad_norm_(actor.actor_parameters() + list(critic.parameters()), max_grad_norm)
        policy_optimizer.step()

        auxiliary = convention_coef * conv_ce + goal_coef * goal_ce
        if belief_optimizer is not None:
            belief_optimizer.zero_grad()
            auxiliary.backward()
            nn.utils.clip_grad_norm_(actor.belief_parameters(), max_grad_norm)
            belief_optimizer.step()
    return {
        'policy_loss': float(policy_loss.item()),
        'value_loss': float(value_loss.item()),
        'convention_ce': float(conv_ce.item()),
        'goal_ce': float(goal_ce.item()),
    }


def train(experiment='experiment2', timesteps=2400, seed=20, learning_rate=2.5e-4,
          episodes_per_update=8, update_epochs=4, convention_coef=1.0,
          goal_coef=1.0, output_dir='results/joint_belief', device='cpu', log_interval=20,
          pretrained_belief=None, freeze_belief=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    n = {'experiment1': 3, 'experiment2': 6}[experiment]
    actor, critic = build_models(experiment, device)
    if pretrained_belief is not None:
        pre = torch.load(pretrained_belief, map_location=device, weights_only=True)
        if pre.get('experiment') != experiment:
            raise ValueError('Pretrained belief uses another experiment')
        actor.belief_gru.load_state_dict(pre['belief_gru'])
        actor.convention_head.load_state_dict(pre['convention_head'])
        actor.goal_head.load_state_dict(pre['goal_head'])
    if freeze_belief:
        if pretrained_belief is None:
            raise ValueError('Cannot freeze a randomly initialized belief model')
        for parameter in actor.belief_parameters():
            parameter.requires_grad_(False)
    policy_optimizer = torch.optim.Adam(actor.actor_parameters() + list(critic.parameters()), lr=learning_rate, eps=1e-5)
    belief_optimizer = (torch.optim.Adam(actor.belief_parameters(), lr=learning_rate, eps=1e-5)
                        if not freeze_belief else None)
    env = make_env(experiment)
    recent = deque(maxlen=100)
    completed = successful = steps = updates = 0
    try:
        while steps < timesteps:
            batch = []
            for _ in range(episodes_per_update):
                if steps >= timesteps: break
                ep = collect_episode(env, actor, critic, n, device, seed + completed)
                batch.append(ep)
                recent.append(ep)
                steps += ep['length']
                completed += 1
                successful += int(ep['success'])
            info = update_models(actor, critic, policy_optimizer, belief_optimizer, batch,
                                 epochs=update_epochs, convention_coef=convention_coef, goal_coef=goal_coef)
            updates += 1
            if updates % log_interval == 0 or steps >= timesteps:
                recent_goal = 100 * sum(x['success'] for x in recent) / len(recent)
                recent_id = 100 * sum(x['id_correct_final'] for x in recent) / len(recent)
                recent_gacc = 100 * sum(x['goal_correct_final'] for x in recent) / len(recent)
                print(f'Steps {steps:>7} | Episodes {completed:>6} | Success {100*successful/completed:5.1f}% '
                      f'| Recent success {recent_goal:5.1f}% | Convention ID {recent_id:5.1f}% '
                      f'| Goal ID {recent_gacc:5.1f}% | CE convention {info["convention_ce"]:.3f} '
                      f'| CE goal {info["goal_ce"]:.3f} | PPO {info["policy_loss"]:.3f}', flush=True)
    finally:
        env.close()
    variant = ('joint_pretrained_frozen' if pretrained_belief and freeze_belief
               else 'joint_pretrained_tuned' if pretrained_belief
               else 'joint_scratch')
    path = Path(output_dir) / experiment / f'{variant}_seed_{seed}_steps{steps}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(),
                'optimizer': policy_optimizer.state_dict(), 'belief_optimizer': (None if belief_optimizer is None else belief_optimizer.state_dict()),
                'architecture': 'joint_belief_v1', 'experiment': experiment,
                'agent': 'explicit_joint', 'num_conventions': n, 'num_goals': 8,
                'seed': seed, 'timesteps': steps, 'learning_rate': learning_rate,
                'convention_coef': convention_coef, 'goal_coef': goal_coef,
                'pretrained_belief': pretrained_belief, 'freeze_belief': freeze_belief}, path)
    print('Saved:', path, flush=True)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', choices=['experiment1', 'experiment2'], default='experiment2')
    p.add_argument('--timesteps', type=int, default=2400)
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--learning-rate', type=float, default=2.5e-4)
    p.add_argument('--episodes-per-update', type=int, default=8)
    p.add_argument('--update-epochs', type=int, default=4)
    p.add_argument('--convention-coef', type=float, default=1.0)
    p.add_argument('--goal-coef', type=float, default=1.0)
    p.add_argument('--output-dir', default='results/joint_belief')
    p.add_argument('--device', default='cpu')
    p.add_argument('--log-interval', type=int, default=20)
    p.add_argument('--pretrained-belief', default=None)
    p.add_argument('--freeze-belief', action='store_true')
    a = p.parse_args()
    if min(a.timesteps, a.episodes_per_update, a.update_epochs, a.log_interval) <= 0:
        p.error('All counts must be positive')
    train(experiment=a.experiment, timesteps=a.timesteps, seed=a.seed,
          learning_rate=a.learning_rate, episodes_per_update=a.episodes_per_update,
          update_epochs=a.update_epochs, convention_coef=a.convention_coef,
          goal_coef=a.goal_coef, output_dir=a.output_dir,
          device=a.device, log_interval=a.log_interval,
          pretrained_belief=a.pretrained_belief, freeze_belief=a.freeze_belief)

if __name__ == '__main__':
    main()
