"""Check whether the explicit actor changes its actions when beliefs change.

Run from the project root:
    python -m scripts.test_belief_sensitivity --experiment experiment1
    python -m scripts.test_belief_sensitivity --experiment experiment2

This only reads a checkpoint. It does not train or change the agent.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation.evaluate_mappo import load_actor
from configs.experiments import make_env


def action_probs(actor, local_obs, belief, hidden):
    """Compare one-step beliefs using the same pre-step policy memory."""
    values = torch.cat((local_obs, belief), dim=-1)
    logits, _ = actor.actor_net(values.unsqueeze(1), hidden)
    return torch.softmax(logits[:, -1, :], dim=-1)


def compare_beliefs(actor, experiment, episodes=100, seed_start=10000):
    """Compare different beliefs on the SAME observations.

    Episodes follow the learned-belief policy. Alternative beliefs are
    tested on each visited observation without changing the episode.
    """
    env = make_env(experiment)
    device = next(actor.parameters()).device
    n = actor.num_conventions
    uniform = actor.initial_belief(1, device)
    cases = [f"certain_C{i}" for i in range(n)]
    names = ["uniform"] + cases
    changes = {name: 0 for name in names}
    total_differences = {name: 0.0 for name in names}
    total_kl = {name: 0.0 for name in names}
    observations_checked = 0
    belief_distance = 0.0
    max_changes = 0

    try:
        with torch.no_grad():
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed_start + episode)
                belief = actor.initial_memory(1, device)
                finished = False
                while not finished:
                    local = torch.as_tensor(
                        np.asarray(observation["learner"], dtype=np.float32),
                        device=device,
                    ).unsqueeze(0)

                    # First update the belief using only what the learner sees.
                    _, _, next_memory = actor(local, belief)
                    next_belief = next_memory[0]
                    real_probs = action_probs(actor, local, next_belief, belief[2])
                    real_action = int(real_probs.argmax(dim=-1).item())
                    belief_distance += float((next_belief - uniform).abs().sum().item())

                    alternatives = {"uniform": uniform}
                    for i, name in enumerate(cases):
                        certain = torch.zeros_like(uniform)
                        certain[0, i] = 1.0
                        alternatives[name] = certain

                    actions_changed_this_step = 0
                    for name, alternative in alternatives.items():
                        other_probs = action_probs(actor, local, alternative, belief[2])
                        other_action = int(other_probs.argmax(dim=-1).item())
                        changed = other_action != real_action
                        changes[name] += int(changed)
                        actions_changed_this_step += int(changed)
                        # Total variation distance ranges from zero to one.
                        total_differences[name] += float(
                            0.5 * (real_probs - other_probs).abs().sum().item()
                        )
                        # KL measures the difference between two distributions.
                        total_kl[name] += float(
                            (real_probs * (real_probs.clamp_min(1e-12).log()
                                           - other_probs.clamp_min(1e-12).log())).sum().item()
                        )
                    max_changes = max(max_changes, actions_changed_this_step)
                    observations_checked += 1

                    # Keep following the REAL belief policy in the environment.
                    observation, _, terminated, truncated, _ = env.step(real_action)
                    finished = bool(terminated or truncated)
                    belief = next_memory
    finally:
        env.close()

    if observations_checked == 0:
        raise RuntimeError("No observations were collected")

    print(f"Experiment: {experiment} | Episodes: {episodes}")
    print(f"Observations checked: {observations_checked}")
    print(f"Mean L1 distance of learned beliefs from uniform: {belief_distance / observations_checked:.6f}")
    print("All comparisons use exactly the same observations.")
    print("Alternative beliefs do NOT change the rollout trajectory.\n")
    print(f"{'Belief used':<16} {'Action changes':>15} {'Change %':>10} {'Mean TV':>12} {'Mean KL':>12}")
    for name in names:
        print(f"{name:<16} {changes[name]:>15} "
              f"{100 * changes[name] / observations_checked:>9.2f}% "
              f"{total_differences[name] / observations_checked:>12.6f} "
              f"{total_kl[name] / observations_checked:>12.6f}")
    print("\nInterpretation:")
    print("- Low change % and low TV: action choices are insensitive to the tested beliefs.")
    print("- High TV but low change %: probabilities change without changing the top action.")
    print("- If learned beliefs are almost uniform, the real/uniform comparison is limited.")
    print("- Extreme beliefs test sensitivity; they are not necessarily realistic beliefs.")


def main():
    parser = argparse.ArgumentParser(description="Measure explicit actor belief sensitivity")
    parser.add_argument("--experiment", choices=["experiment1", "experiment2"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    checkpoint = args.checkpoint or str(Path("results") / args.experiment / "explicit" / "seed_1.pt")
    actor = load_actor(checkpoint, args.experiment, "explicit", args.device)
    compare_beliefs(actor, args.experiment, args.episodes, args.seed_start)


if __name__ == "__main__":
    main()
