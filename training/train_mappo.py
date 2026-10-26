
"""Train the implicit GRU or explicit belief agent with recurrent PPO.

Run from the project root:
python -m training.train_mappo --experiment experiment1 --agent implicit --timesteps 2400
python -m training.train_mappo --experiment experiment2 --agent explicit --timesteps 2400

Each training batch contains complete episodes (up to 12 steps). This keeps
GRU memory and belief updates in the correct time order during PPO updates.
"""

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from agents.centralized_critic import CentralizedCritic, build_critic_state
from agents.explicit_belief_agent import ExplicitBeliefActor
from agents.implicit_gru_agent import GRUActor


def create_models(agent_name, num_conventions, device):
    if agent_name == "implicit":
        actor = GRUActor(obs_dim=5, action_dim=5, hidden_dim=64)
    elif agent_name == "explicit":
        actor = ExplicitBeliefActor(obs_dim=5, action_dim=5,
                                    num_conventions=num_conventions, hidden_dim=64)
    else:
        raise ValueError("agent must be 'implicit' or 'explicit'")
    critic = CentralizedCritic(num_conventions=num_conventions, hidden_dim=64)
    return actor.to(device), critic.to(device)


def discounted_gae(rewards, values, next_value, terminated, gamma=0.99, gae_lambda=0.95):
    """Compute advantages for one complete episode.

    On success, there is no future value. On a time-limit truncation,
    bootstrap from the observation returned by the final environment step.
    """
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(len(rewards))):
        value_after = (torch.zeros_like(next_value) if terminated else next_value) if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * value_after - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    return advantages, advantages + values


def collect_episode(env, actor, critic, agent_name, num_conventions, device, seed=None,
                    gamma=0.99, gae_lambda=0.95):
    """Collect one entire episode, and reset the actor memory at its start."""
    observation, _ = env.reset(seed=seed)
    observations, states, actions, logps, rewards = [], [], [], [], []
    convention = int(observation["convention"][0])
    memory = actor.initial_hidden(1, device) if agent_name == "implicit" else actor.initial_belief(1, device)
    terminated = truncated = False
    id_correct_steps = []

    with torch.no_grad():
        while not (terminated or truncated):
            state = build_critic_state(observation, len(rewards), max_steps=env.env.max_steps,
                                       num_conventions=num_conventions)
            local_obs = np.asarray(observation["learner"], dtype=np.float32)
            obs_tensor = torch.as_tensor(local_obs, device=device).unsqueeze(0)
            action, logp, _, memory = actor.get_action(obs_tensor, memory)
            action_int = int(action.item())
            if agent_name == "explicit":
                # This is the predicted belief, never the true convention.
                id_correct_steps.append(int(memory.argmax(dim=-1).item() == convention))
            observations.append(local_obs)
            states.append(state)
            actions.append(action_int)
            logps.append(float(logp.item()))
            observation, reward, terminated, truncated, _ = env.step(action_int)
            rewards.append(float(reward))

        # A truncation cuts off the episode, but not the value estimate.
        final_state = build_critic_state(observation, len(rewards),
                                         max_steps=env.env.max_steps,
                                         num_conventions=num_conventions)
        next_value = critic(torch.as_tensor(final_state, device=device))
        states_tensor = torch.as_tensor(np.asarray(states), device=device)
        values = critic(states_tensor)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        advantages, returns = discounted_gae(reward_tensor, values, next_value,
                                             terminated, gamma, gae_lambda)
    return {
        "observations": torch.as_tensor(np.asarray(observations), device=device),
        "states": states_tensor,
        "actions": torch.tensor(actions, dtype=torch.long, device=device),
        "old_logps": torch.tensor(logps, dtype=torch.float32, device=device),
        "advantages": advantages,
        "returns": returns,
        "convention": convention,
        "success": bool(terminated),
        "reward": sum(rewards),
        "length": len(rewards),
        "truncated": bool(truncated),
        "id_correct_final": id_correct_steps[-1] if id_correct_steps else None,
        "id_correct_steps": id_correct_steps,
    }


def update_models(actor, critic, optimizer, episodes, agent_name, *, epochs=4,
                  clip_coef=0.2, value_coef=0.5, entropy_coef=0.01, ce_coef=0.1,
                  max_grad_norm=0.5, belief_optimizer=None):
    """Update using full episodes, so recurrent history is never shuffled."""
    if agent_name == "explicit" and belief_optimizer is None:
        raise ValueError("The explicit belief network needs its own optimizer")
    all_adv = torch.cat([ep["advantages"] for ep in episodes])
    adv_mean = all_adv.mean()
    adv_std = all_adv.std(unbiased=False).clamp_min(1e-8)
    summary = {}
    for _ in range(epochs):
        # Recompute the entire sequence with current actor weights each epoch.
        # A single update uses all collected episodes; gradient sizes are
        # weighted by the number of steps instead of episode length.
        total_steps = sum(ep["length"] for ep in episodes)
        policy_loss = value_loss = entropy_loss = belief_loss = None
        for ep in episodes:
            obs = ep["observations"].unsqueeze(0)
            acts = ep["actions"].unsqueeze(0)
            if agent_name == "implicit":
                logp, entropy, _ = actor.evaluate_actions(obs, acts)
                ce = None
            else:
                logp, entropy, belief_logits, _ = actor.evaluate_actions(obs, acts)
                targets = torch.full((ep["length"],), ep["convention"],
                                     device=obs.device, dtype=torch.long)
                ce = F.cross_entropy(belief_logits.squeeze(0), targets, reduction="sum")
            logp = logp.squeeze(0)
            entropy = entropy.squeeze(0)
            ratios = (logp - ep["old_logps"]).exp()
            advantage = (ep["advantages"] - adv_mean) / adv_std
            unclipped = ratios * advantage
            clipped = ratios.clamp(1 - clip_coef, 1 + clip_coef) * advantage
            p = -torch.minimum(unclipped, clipped).sum() / total_steps
            predicted_values = critic(ep["states"])
            v = (predicted_values - ep["returns"]).square().sum() / (2 * total_steps)
            e = entropy.sum() / total_steps
            policy_loss = p if policy_loss is None else policy_loss + p
            value_loss = v if value_loss is None else value_loss + v
            entropy_loss = e if entropy_loss is None else entropy_loss + e
            if ce is not None:
                c = ce / total_steps
                belief_loss = c if belief_loss is None else belief_loss + c

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        policy_parameters = (actor.actor_net.parameters() if agent_name == "explicit"
                             else actor.parameters())
        nn.utils.clip_grad_norm_(list(policy_parameters) + list(critic.parameters()), max_grad_norm)
        optimizer.step()
        if belief_loss is not None:
            # Identification alone trains the standalone belief network.
            belief_optimizer.zero_grad()
            (ce_coef * belief_loss).backward()
            nn.utils.clip_grad_norm_(actor.belief_net.parameters(), max_grad_norm)
            belief_optimizer.step()
        summary = {"policy_loss": float(policy_loss.item()),
                   "value_loss": float(value_loss.item()),
                   "belief_ce": None if belief_loss is None else float(belief_loss.item())}
    return summary


def train(experiment="experiment1", agent_name="implicit", timesteps=2400, seed=1,
          learning_rate=2.5e-4, episodes_per_update=8, update_epochs=4,
          ce_coef=0.1, output_dir="results", device="cpu", env_factory=None):
    """Train from scratch and save an actor + critic checkpoint."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_conventions = {"experiment1": 3, "experiment2": 6}[experiment]
    if env_factory is None:
        from configs.experiments import make_env
        env_factory = make_env
    env = env_factory(experiment)
    device = torch.device(device)
    actor, critic = create_models(agent_name, num_conventions, device)
    policy_parameters = (actor.actor_net.parameters() if agent_name == "explicit"
                         else actor.parameters())
    optimizer = torch.optim.Adam(list(policy_parameters) + list(critic.parameters()),
                                 lr=learning_rate, eps=1e-5)
    belief_optimizer = (torch.optim.Adam(actor.belief_net.parameters(),
                                        lr=learning_rate, eps=1e-5)
                        if agent_name == "explicit" else None)
    steps = completed = successful = 0
    recent = deque(maxlen=100)
    last_info = {}
    try:
        while steps < timesteps:
            batch = []
            for _ in range(episodes_per_update):
                if steps >= timesteps:
                    break
                # A new seed for every episode gives reproducible variation.
                episode = collect_episode(env, actor, critic, agent_name, num_conventions,
                                          device, seed=seed + completed)
                batch.append(episode)
                steps += episode["length"]
                completed += 1
                successful += int(episode["success"])
                recent.append(episode)
            last_info = update_models(actor, critic, optimizer, batch, agent_name,
                                      epochs=update_epochs, ce_coef=ce_coef,
                                      belief_optimizer=belief_optimizer)
            # Recent metrics use at most the last 100 finished episodes.
            recent_success = 100 * sum(ep["success"] for ep in recent) / len(recent)
            message = (f"Steps {steps:>7} | Episodes {completed:>5} | "
                       f"Goal success {100*successful/completed:5.1f}% | "
                       f"Recent goal success {recent_success:5.1f}% | "
                       f"PPO loss {last_info['policy_loss']:.3f}")
            if agent_name == "explicit":
                # Final ID: the last convention prediction in each episode.
                final_accuracy = 100 * sum(ep["id_correct_final"] for ep in recent) / len(recent)
                # Every prediction made during the recent episodes counts once.
                step_hits = sum(sum(ep["id_correct_steps"]) for ep in recent)
                step_count = sum(len(ep["id_correct_steps"]) for ep in recent)
                step_accuracy = 100 * step_hits / step_count
                message += (f" | Recent ID final {final_accuracy:5.1f}%"
                            f" | Recent ID steps {step_accuracy:5.1f}%"
                            f" | CE {last_info['belief_ce']:.3f}")
            else:
                message += " | CE N/A"
            print(message)
    finally:
        env.close()
    destination = Path(output_dir) / experiment / agent_name / f"seed_{seed}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"actor": actor.state_dict(), "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(), "experiment": experiment,
                "agent": agent_name, "num_conventions": num_conventions,
                "seed": seed, "timesteps": steps, "episodes": completed,
                "goal_success": successful / completed,
                "learning_rate": learning_rate, "ce_coef": ce_coef}
    if belief_optimizer is not None:
        checkpoint["belief_optimizer"] = belief_optimizer.state_dict()
    torch.save(checkpoint, destination)
    print(f"Saved: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Recurrent PPO training for GridWorld")
    parser.add_argument("--experiment", choices=["experiment1", "experiment2"], required=True)
    parser.add_argument("--agent", choices=["implicit", "explicit"], required=True)
    parser.add_argument("--timesteps", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--ce-coef", type=float, default=0.1)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.episodes_per_update <= 0 or args.update_epochs <= 0:
        parser.error("timesteps, episodes-per-update and update-epochs must be positive")
    train(experiment=args.experiment, agent_name=args.agent, timesteps=args.timesteps,
          seed=args.seed, learning_rate=args.learning_rate,
          episodes_per_update=args.episodes_per_update,
          update_epochs=args.update_epochs, ce_coef=args.ce_coef,
          output_dir=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
