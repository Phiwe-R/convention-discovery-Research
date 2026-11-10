
"""Clean goal-guided PPO: no oracle-action imitation; CSV + TensorBoard logging."""
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

from agents.centralized_critic import CentralizedCritic
from agents.goal_guided_actor import GoalGuidedJointActor
from configs.experiments import make_env
from training.train_joint_belief import collect_episode

FIELDS = [
    "experiment", "seed", "target_timesteps", "step", "update", "episodes",
    "cumulative_avg_reward", "recent_avg_reward",
    "cumulative_goal_success_pct", "recent_goal_success_pct",
    "cumulative_convention_id_step_pct", "recent_convention_id_step_pct",
    "cumulative_convention_id_episode_pct", "recent_convention_id_episode_pct",
    "cumulative_goal_id_step_pct", "recent_goal_id_step_pct",
    "cumulative_goal_id_episode_pct", "recent_goal_id_episode_pct",
    "policy_loss", "value_loss", "policy_entropy", "entropy_loss",
    "belief_convention_ce", "belief_goal_ce", "belief_total_ce",
    "ppo_total_loss", "supervised_action_cross_entropy_loss", "learning_rate",
]


def percent(n, d):
    return 100.0 * n / d if d else 0.0


def summary(episodes):
    steps = sum(e["length"] for e in episodes)
    return dict(
        avg_reward=sum(e["reward"] for e in episodes) / max(len(episodes), 1),
        goal_success_pct=percent(sum(e["success"] for e in episodes), len(episodes)),
        convention_id_step_pct=percent(sum(sum(e["id_correct_steps"]) for e in episodes), steps),
        convention_id_episode_pct=percent(sum(e["id_correct_final"] for e in episodes), len(episodes)),
        goal_id_step_pct=percent(sum(sum(e["goal_correct_steps"]) for e in episodes), steps),
        goal_id_episode_pct=percent(sum(e["goal_correct_final"] for e in episodes), len(episodes)),
    )


def ppo_update(actor, critic, optimizer, episodes, epochs=4):
    """Ground-truth labels are used ONLY to report frozen-belief CE."""
    all_adv = torch.cat([e["advantages"] for e in episodes])
    mu = all_adv.mean()
    sigma = all_adv.std(unbiased=False).clamp_min(1e-8)
    total_steps = sum(e["length"] for e in episodes)
    logged = []

    for _ in range(epochs):
        policy = value = entropy = 0.0

        for ep in episodes:
            obs = ep["observations"].unsqueeze(0)
            logits, _, _, _ = actor(obs)

            new_logp = F.log_softmax(logits, -1).gather(
                -1, ep["actions"][None, :, None]
            ).squeeze(0).squeeze(-1)

            ratio = (new_logp - ep["old_logps"]).exp()
            advantage = (ep["advantages"] - mu) / sigma

            policy = policy - torch.minimum(
                ratio * advantage,
                ratio.clamp(0.8, 1.2) * advantage,
            ).sum() / total_steps

            value = value + (
                critic(ep["states"]) - ep["returns"]
            ).square().sum() / (2 * total_steps)

            probs = logits.softmax(-1)
            entropy = entropy - (
                probs * F.log_softmax(logits, -1)
            ).sum() / total_steps

        entropy_term = -0.01 * entropy

        # PPO ONLY: no true-goal imitation objective.
        total = policy + 0.5 * value + entropy_term

        optimizer.zero_grad(set_to_none=True)
        total.backward()

        clip_grad_norm_(
            actor.actor_parameters() + list(critic.parameters()),
            0.5,
        )
        optimizer.step()

        logged.append([
            float(x.detach())
            for x in (policy, value, entropy, entropy_term, total)
        ])

    # Diagnostic CE only; never used for action selection or PPO gradients.
    conv_sum = goal_sum = 0.0
    actor.eval()

    with torch.no_grad():
        for ep in episodes:
            obs = ep["observations"].unsqueeze(0)
            _, conv_logits, goal_logits, _ = actor(obs)

            conv_target = torch.full(
                (ep["length"],),
                int(ep["convention"]),
                dtype=torch.long,
                device=obs.device,
            )
            goal_target = torch.full(
                (ep["length"],),
                int(ep["goal_id"]),
                dtype=torch.long,
                device=obs.device,
            )

            conv_sum += float(
                F.cross_entropy(
                    conv_logits.squeeze(0),
                    conv_target,
                    reduction="sum",
                )
            )
            goal_sum += float(
                F.cross_entropy(
                    goal_logits.squeeze(0),
                    goal_target,
                    reduction="sum",
                )
            )

    actor.train()

    policy, value, entropy, entropy_term, total = np.asarray(logged).mean(axis=0)

    return dict(
        policy_loss=float(policy),
        value_loss=float(value),
        policy_entropy=float(entropy),
        entropy_loss=float(entropy_term),
        ppo_total_loss=float(total),
        belief_convention_ce=conv_sum / total_steps,
        belief_goal_ce=goal_sum / total_steps,
        belief_total_ce=(conv_sum + goal_sum) / total_steps,
        supervised_action_cross_entropy_loss=0.0,
    )


def train(
    experiment,
    timesteps,
    seed,
    pretrained_belief,
    guide_strength=5.0,
    learning_rate=2.5e-4,
    episodes_per_update=8,
    update_epochs=4,
    recent_window=100,
    print_interval=20,
    device="cpu",
    output_dir="results/goal_guided_clean",
    csv_root="results/goal_guided_clean/training_logs",
    tensorboard_root="runs/goal_guided_clean",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "cpu":
        torch.set_num_threads(1)

    device = torch.device(device)

    n = {
        "experiment1": 3,
        "experiment2": 6,
    }[experiment]

    actor = GoalGuidedJointActor(
        num_conventions=n,
        guide_strength=guide_strength,
    ).to(device)

    ckpt = torch.load(
        pretrained_belief,
        map_location=device,
        weights_only=True,
    )

    if ckpt.get("experiment") != experiment:
        raise ValueError(
            "Belief checkpoint is for a different experiment"
        )

    for name in (
        "belief_gru",
        "convention_head",
        "goal_head",
    ):
        getattr(actor, name).load_state_dict(ckpt[name])

    # Freeze belief network during PPO.
    for p in actor.belief_parameters():
        p.requires_grad_(False)

    critic = CentralizedCritic(
        num_conventions=n,
    ).to(device)

    optimizer = torch.optim.Adam(
        actor.actor_parameters() + list(critic.parameters()),
        lr=learning_rate,
        eps=1e-5,
    )

    run = f"seed{seed}_target{timesteps}"

    csv_dir = Path(csv_root) / experiment / run
    tb_dir = Path(tensorboard_root) / experiment / run

    csv_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / "training_metrics.csv"

    (csv_dir / "run_config.json").write_text(
        json.dumps({
            "experiment": experiment,
            "seed": seed,
            "budget": timesteps,
            "pretrained_belief": str(pretrained_belief),
            "guide_strength": guide_strength,
            "imitation_enabled": False,
            "belief_frozen": True,
        }, indent=2),
        encoding="utf-8",
    )

    writer = SummaryWriter(str(tb_dir))
    env = make_env(experiment)

    recent = deque(maxlen=recent_window)

    count = 0
    hits = 0
    reward_sum = 0
    conv_steps = 0
    conv_final = 0
    goal_steps = 0
    goal_final = 0

    steps = 0
    updates = 0

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        csv_writer = csv.DictWriter(
            fh,
            fieldnames=FIELDS,
        )
        csv_writer.writeheader()

        try:
            while steps < timesteps:
                batch = []

                for _ in range(episodes_per_update):
                    if steps >= timesteps:
                        break

                    ep = collect_episode(
                        env,
                        actor,
                        critic,
                        n,
                        device,
                        seed + count,
                    )

                    batch.append(ep)
                    recent.append(ep)

                    count += 1
                    hits += int(ep["success"])
                    reward_sum += ep["reward"]

                    conv_steps += sum(ep["id_correct_steps"])
                    conv_final += ep["id_correct_final"]
                    goal_steps += sum(ep["goal_correct_steps"])
                    goal_final += ep["goal_correct_final"]

                    steps += ep["length"]

                if not batch:
                    break

                losses = ppo_update(
                    actor,
                    critic,
                    optimizer,
                    batch,
                    epochs=update_epochs,
                )

                updates += 1

                row = {
                    "experiment": experiment,
                    "seed": seed,
                    "target_timesteps": timesteps,
                    "step": steps,
                    "update": updates,
                    "episodes": count,

                    "cumulative_avg_reward": reward_sum / count,
                    "cumulative_goal_success_pct": percent(hits, count),

                    "cumulative_convention_id_step_pct": percent(
                        conv_steps, steps
                    ),
                    "cumulative_convention_id_episode_pct": percent(
                        conv_final, count
                    ),
                    "cumulative_goal_id_step_pct": percent(
                        goal_steps, steps
                    ),
                    "cumulative_goal_id_episode_pct": percent(
                        goal_final, count
                    ),

                    **{
                        f"recent_{k}": v
                        for k, v in summary(recent).items()
                    },

                    **losses,

                    "learning_rate": optimizer.param_groups[0]["lr"],
                }

                # Save CSV row every PPO update.
                csv_writer.writerow(row)
                fh.flush()

                # Save TensorBoard scalars every PPO update.
                for key, value in row.items():
                    if key not in (
                        "experiment",
                        "seed",
                        "target_timesteps",
                        "step",
                        "update",
                    ):
                        writer.add_scalar(
                            key,
                            value,
                            steps,
                        )

                writer.flush()

                if (
                    updates % print_interval == 0
                    or steps >= timesteps
                ):
                    print(
                        f"Steps {steps:>7} | Episodes {count:>6} "
                        f"| Success {row['cumulative_goal_success_pct']:.1f}% "
                        f"| Recent {row['recent_goal_success_pct']:.1f}% "
                        f"| Reward {row['recent_avg_reward']:.3f} "
                        f"| Conv-step {row['recent_convention_id_step_pct']:.1f}% "
                        f"| Conv-episode {row['recent_convention_id_episode_pct']:.1f}% "
                        f"| Goal ID {row['recent_goal_id_episode_pct']:.1f}% "
                        f"| Policy {row['policy_loss']:.3f} "
                        f"| Conv CE {row['belief_convention_ce']:.3f} "
                        f"| Goal CE {row['belief_goal_ce']:.3f}",
                        flush=True,
                    )

        finally:
            env.close()
            writer.close()

    # Save separately from imitation-trained checkpoints.
    path = (
        Path(output_dir)
        / experiment
        / f"guided_clean_seed_{seed}_steps{steps}.pt"
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "architecture": "joint_goal_guided_clean_v1",
        "experiment": experiment,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": seed,
        "timesteps": steps,
        "target_timesteps": timesteps,
        "num_conventions": n,
        "num_goals": 8,
        "guide_strength": guide_strength,
        "pretrained_belief": str(pretrained_belief),
        "imitation_loss_enabled": False,
        "training_csv": str(csv_path),
        "tensorboard_dir": str(tb_dir),
    }, path)

    print(
        "Saved:", path,
        "\nCSV:", csv_path,
        "\nTensorBoard:", tb_dir,
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--experiment",
        choices=["experiment1", "experiment2"],
        required=True,
    )
    p.add_argument("--timesteps", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pretrained-belief", required=True)
    p.add_argument("--guide-strength", type=float, default=5.0)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--recent-window", type=int, default=100)
    p.add_argument("--print-interval", type=int, default=20)
    p.add_argument("--device", default="cpu")

    train(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
