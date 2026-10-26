"""Evaluate saved MAPPO agents on the same randomly seeded episodes.

Run from the research project's root folder, for example:
python -m evaluation.evaluate_mappo --experiment experiment1 --agent implicit --episodes 100
python -m evaluation.evaluate_mappo --experiment experiment1 --agent explicit --episodes 100
python -m evaluation.evaluate_mappo --experiment experiment1 --agent explicit --episodes 100 --belief-mode uniform

The evaluation never updates network weights.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from training.train_mappo import create_models


def load_actor(checkpoint_path, experiment, agent_name, device):
    """Load the trained actor. The critic is not needed for choosing actions."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected = {"experiment1": 3, "experiment2": 6}[experiment]
    if checkpoint.get("experiment") != experiment or checkpoint.get("agent") != agent_name:
        raise ValueError("Checkpoint agent or experiment does not match the command")
    if checkpoint.get("num_conventions") != expected:
        raise ValueError("Checkpoint has the wrong number of conventions")
    actor, _ = create_models(agent_name, expected, torch.device(device))
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def run_episode(env, actor, agent_name, seed, belief_mode="real", deterministic=True):
    """Evaluate one episode. Hidden labels are used only for scoring."""
    if belief_mode != "real" and agent_name != "explicit":
        raise ValueError("Uniform-belief testing is only for the explicit agent")
    observation, info = env.reset(seed=seed)
    # Read the true label for evaluation, never for actor input.
    true_convention = int(observation["convention"][0])
    device = next(actor.parameters()).device
    memory = (actor.initial_hidden(1, device) if agent_name == "implicit"
              else actor.initial_memory(1, device))
    reward_sum = 0.0
    matching = []
    done = truncated = False
    steps = 0
    with torch.no_grad():
        while not (done or truncated):
            local = torch.as_tensor(np.asarray(observation["learner"], dtype=np.float32),
                                    device=device).unsqueeze(0)
            if agent_name == "explicit" and belief_mode == "uniform":
                # Ablate beliefs while preserving the action GRU's own history.
                uniform = actor.initial_belief(1, device)
                policy_input = torch.cat((local, uniform), dim=-1).unsqueeze(1)
                logits, action_hidden = actor.actor_net(policy_input, memory[2])
                logits = logits[:, -1, :]
                memory = (uniform, memory[1], action_hidden)
                action = logits.argmax(dim=-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
            else:
                action, _, _, memory = actor.get_action(local, memory, deterministic=deterministic)
                if agent_name == "explicit":
                    prediction = int(memory[0].argmax(dim=-1).item())
                    matching.append(int(prediction == true_convention))
            observation, reward, done, truncated, info = env.step(int(action.item()))
            reward_sum += float(reward)
            steps += 1
            if steps > 10000:
                raise RuntimeError("Environment did not terminate")
    return {
        "seed": seed,
        "goal_success": int(bool(info.get("task_success", done))),
        "reward": reward_sum,
        "length": steps,
        "truncated": int(bool(truncated)),
        "convention": true_convention,
        "id_correct_last": matching[-1] if matching else None,
        "id_by_step": matching,
    }


def evaluate(actor, env_factory, experiment, agent_name, episodes=100,
             seed_start=10000, belief_mode="real", deterministic=True):
    """Reset with the same seeds for fair comparisons."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = env_factory(experiment)
    try:
        records = [run_episode(env, actor, agent_name, seed_start + i,
                               belief_mode, deterministic) for i in range(episodes)]
    finally:
        env.close()
    summary = {
        "episodes": len(records),
        "goal_success_pct": 100 * np.mean([r["goal_success"] for r in records]),
        "mean_reward": float(np.mean([r["reward"] for r in records])),
        "mean_length": float(np.mean([r["length"] for r in records])),
        "truncated_pct": 100 * np.mean([r["truncated"] for r in records]),
    }
    if agent_name == "explicit" and belief_mode == "real":
        summary["id_accuracy_final_pct"] = 100 * np.mean([r["id_correct_last"] for r in records])
        # Only count episodes with an actual prediction at this step.
        summary["id_accuracy_by_step_pct"] = {
            t + 1: 100 * float(np.mean([r["id_by_step"][t] for r in records if len(r["id_by_step"]) > t]))
            for t in range(max(len(r["id_by_step"]) for r in records))
        }
    return summary, records


def save_records(path, records):
    """Save one row per episode for later statistical comparisons."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "seed", "goal_success", "reward", "length", "truncated", "convention", "id_correct_last"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow({key: r[key] for key in writer.fieldnames})
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate a MAPPO checkpoint")
    parser.add_argument("--experiment", choices=["experiment1", "experiment2"], required=True)
    parser.add_argument("--agent", choices=["implicit", "explicit"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--belief-mode", choices=["real", "uniform"], default="real")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of picking the most likely one")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    from configs.experiments import make_env
    checkpoint_path = args.checkpoint or str(Path("results") / args.experiment / args.agent / "seed_1.pt")
    actor = load_actor(checkpoint_path, args.experiment, args.agent, args.device)
    summary, records = evaluate(actor, make_env, args.experiment, args.agent,
                                episodes=args.episodes, seed_start=args.seed_start,
                                belief_mode=args.belief_mode,
                                deterministic=not args.stochastic)
    if args.agent == "implicit":
        print(
            f"Agent: {args.agent} | "
            f"Experiment: {args.experiment} | "
            f"Beliefs: N/A (GRU hidden memory)"
        )
    else:
        print(
            f"Agent: {args.agent} | "
            f"Experiment: {args.experiment} | "
            f"Beliefs: {args.belief_mode}"
        )
    for key, value in summary.items():
        if isinstance(value, dict):
            print("Identification accuracy by step (%):", value)
        else:
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    output = args.output or str(Path("results") / args.experiment / args.agent /
                                f"evaluation_{args.belief_mode}.csv")
    save_records(output, records)
    print(f"Saved episode results: {output}")


if __name__ == "__main__":
    main()
