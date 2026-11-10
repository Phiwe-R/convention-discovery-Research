
"""Balanced belief pretraining for C0-C2 (Exp1) or C0-C5 (Exp2)."""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from agents.joint_belief_agent import JointBeliefActor
from configs.experiments import make_env


def make_dataset(experiment, per_convention, seed_offset, seed):
    """Equal convention counts and approximately equal goal pairs."""
    n = 3 if experiment == "experiment1" else 6

    rng = random.Random(seed + seed_offset)
    plan = []

    for convention in range(n):
        pairs = [
            (g, p)
            for g in range(8)
            for p in range(8)
        ]

        remaining = per_convention

        while remaining:
            rng.shuffle(pairs)
            selected = pairs[:min(remaining, len(pairs))]

            plan.extend(
                (convention, g, p)
                for g, p in selected
            )

            remaining -= len(selected)

    rng.shuffle(plan)

    env = make_env(experiment)
    examples = []

    try:
        for index, (c, g, p) in enumerate(plan):
            episode_seed = seed_offset + index

            obs, info = env.reset(
                seed=episode_seed,
                options={
                    "convention_id": c,
                    "learner_goal_id": g,
                    "partner_goal_id": p,
                },
            )

            if (
                int(obs["convention"][0]) != c
                or int(info["learner_goal_id"]) != g
            ):
                raise RuntimeError(
                    "Environment reset did not honour forced labels"
                )

            history = []
            partner_next_actions = []

            policy = index % 3
            local_rng = random.Random(episode_seed + seed)

            ended = False
            truncated = False

            while not (ended or truncated):
                history.append(
                    np.asarray(
                        obs["learner"],
                        dtype=np.float32,
                    )
                )

                if policy == 0:
                    action = 4
                elif policy == 1:
                    action = local_rng.randrange(5)
                else:
                    action = (len(history) + episode_seed) % 5

                obs, _, ended, truncated, _ = env.step(action)

                partner_next_actions.append(
                    int(obs["learner"][4])
                )

            examples.append((
                torch.as_tensor(
                    np.stack(history),
                    dtype=torch.float32,
                ),
                c,
                g,
                torch.tensor(
                    partner_next_actions,
                    dtype=torch.long,
                ),
            ))

    finally:
        env.close()

    return examples


def collate(examples):
    histories, conventions, goals, movements = zip(*examples)

    lengths = torch.tensor([
        len(x) for x in histories
    ])

    return (
        pad_sequence(
            histories,
            batch_first=True,
        ),
        lengths,
        torch.tensor(conventions),
        torch.tensor(goals),
        pad_sequence(
            movements,
            batch_first=True,
            padding_value=-100,
        ),
    )


class PretrainNetwork(nn.Module):
    def __init__(self, num_conventions):
        super().__init__()

        self.belief = JointBeliefActor(
            num_conventions=num_conventions
        )

        self.movement_head = nn.Linear(
            self.belief.hidden_dim,
            5,
        )

    def forward(self, observations):
        features, _ = self.belief.belief_gru(
            observations
        )

        return (
            self.belief.convention_head(features),
            self.belief.goal_head(features),
            self.movement_head(features),
        )


def calculate(model, batch, device):
    obs, lengths, c, g, movements = [
        x.to(device) for x in batch
    ]

    lc, lg, lm = model(obs)

    times = torch.arange(
        obs.shape[1],
        device=device,
    )[None, :]

    mask = times < lengths[:, None]

    c_repeat = c[:, None].expand_as(mask)
    g_repeat = g[:, None].expand_as(mask)

    # Cross entropy across all valid timesteps.
    ce_c = F.cross_entropy(
        lc[mask],
        c_repeat[mask],
    )

    ce_g = F.cross_entropy(
        lg[mask],
        g_repeat[mask],
    )

    ce_m = F.cross_entropy(
        lm[mask],
        movements[mask],
    )

    idx = torch.arange(
        len(lengths),
        device=device,
    )

    # Additional supervision at final timestep.
    final_c = F.cross_entropy(
        lc[idx, lengths - 1],
        c,
    )

    final_g = F.cross_entropy(
        lg[idx, lengths - 1],
        g,
    )

    loss = (
        ce_c
        + ce_g
        + 1.5 * (final_c + final_g)
        + 0.3 * ce_m
    )

    return {
        "loss": loss,
        "convention_ce": ce_c,
        "goal_ce": ce_g,
        "next_action_ce": ce_m,

        "final_convention_pct": 100 * (
            lc[idx, lengths - 1].argmax(-1) == c
        ).float().mean(),

        "final_goal_pct": 100 * (
            lg[idx, lengths - 1].argmax(-1) == g
        ).float().mean(),

        "step_convention_pct": 100 * (
            lc[mask].argmax(-1) == c_repeat[mask]
        ).float().mean(),

        "step_goal_pct": 100 * (
            lg[mask].argmax(-1) == g_repeat[mask]
        ).float().mean(),

        "next_action_pct": 100 * (
            lm[mask].argmax(-1) == movements[mask]
        ).float().mean(),
    }


def validate(model, loader, device):
    model.eval()

    totals = {}
    total_episodes = 0

    with torch.no_grad():
        for batch in loader:
            stats = calculate(
                model,
                batch,
                device,
            )

            n = len(batch[1])
            total_episodes += n

            for name, value in stats.items():
                totals[name] = (
                    totals.get(name, 0.0)
                    + float(value) * n
                )

    return {
        name: value / total_episodes
        for name, value in totals.items()
    }


def timestep_test(model, loader, device):
    counters = {}
    model.eval()

    with torch.no_grad():
        for batch in loader:
            obs, lengths, c, g, _ = [
                x.to(device) for x in batch
            ]

            lc, lg, _ = model(obs)

            for t in range(obs.shape[1]):
                valid = lengths > t

                if not bool(valid.any()):
                    continue

                bucket = counters.setdefault(
                    t + 1,
                    [0, 0, 0],
                )

                bucket[0] += int(valid.sum())

                bucket[1] += int((
                    lc[valid, t].argmax(-1) == c[valid]
                ).sum())

                bucket[2] += int((
                    lg[valid, t].argmax(-1) == g[valid]
                ).sum())

    return [
        {
            "timestep": t,
            "episodes": n,
            "convention_accuracy_pct": 100 * c / n,
            "goal_accuracy_pct": 100 * g / n,
        }
        for t, (n, c, g) in sorted(counters.items())
    ]


def write_rows(path, rows):
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def pretrain(
    experiment,
    output,
    train_per_convention=10000,
    val_per_convention=1000,
    test_per_convention=1000,
    epochs=35,
    batch_size=128,
    patience=6,
    seed=20,
    device="cpu",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "cpu":
        torch.set_num_threads(1)

    device = torch.device(device)

    n = (
        3 if experiment == "experiment1"
        else 6
    )

    print(
        "Generating balanced training episodes...",
        flush=True,
    )

    train_data = make_dataset(
        experiment,
        train_per_convention,
        200000,
        seed,
    )

    print(
        "Generating validation episodes...",
        flush=True,
    )

    val_data = make_dataset(
        experiment,
        val_per_convention,
        400000,
        seed,
    )

    print(
        "Generating test episodes...",
        flush=True,
    )

    test_data = make_dataset(
        experiment,
        test_per_convention,
        800000,
        seed,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=collate,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=collate,
    )

    model = PretrainNetwork(n).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
    )

    path = Path(output)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = SummaryWriter(
        str(
            Path("runs/belief_pretraining")
            / experiment
        )
    )

    rows = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    try:
        for epoch in range(1, epochs + 1):
            model.train()

            for batch in train_loader:
                stats = calculate(
                    model,
                    batch,
                    device,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                stats["loss"].backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    0.5,
                )

                optimizer.step()

            train_stats = validate(
                model,
                train_loader,
                device,
            )

            val_stats = validate(
                model,
                val_loader,
                device,
            )

            row = {"epoch": epoch}

            for split, metrics in (
                ("train", train_stats),
                ("val", val_stats),
            ):
                for key, val in metrics.items():
                    row[f"{split}_{key}"] = val

                    writer.add_scalar(
                        f"{split}/{key}",
                        val,
                        epoch,
                    )

            rows.append(row)
            writer.flush()

            print(
                f"Epoch {epoch:02} "
                f"| Val loss {val_stats['loss']:.3f} "
                f"| Convention ID {val_stats['final_convention_pct']:.1f}% "
                f"| Goal ID {val_stats['final_goal_pct']:.1f}%",
                flush=True,
            )

            if val_stats["loss"] < best_loss - 0.001:
                best_loss = val_stats["loss"]
                best_epoch = epoch

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                stale = 0

            else:
                stale += 1

                if stale >= patience:
                    print(
                        "Early stopping triggered.",
                        flush=True,
                    )
                    break

        model.load_state_dict(best_state)

        test_stats = validate(
            model,
            test_loader,
            device,
        )

        by_step = timestep_test(
            model,
            test_loader,
            device,
        )

        write_rows(
            path.with_name(
                path.stem + "_history.csv"
            ),
            rows,
        )

        write_rows(
            path.with_name(
                path.stem + "_test_by_timestep.csv"
            ),
            by_step,
        )

        for entry in by_step:
            writer.add_scalar(
                "test/convention_by_timestep",
                entry["convention_accuracy_pct"],
                entry["timestep"],
            )

            writer.add_scalar(
                "test/goal_by_timestep",
                entry["goal_accuracy_pct"],
                entry["timestep"],
            )

        torch.save({
            "architecture": "joint_belief_balanced_pretrain_v1",
            "experiment": experiment,
            "num_conventions": n,
            "num_goals": 8,
            "belief_gru": model.belief.belief_gru.state_dict(),
            "convention_head": model.belief.convention_head.state_dict(),
            "goal_head": model.belief.goal_head.state_dict(),
            "movement_head": model.movement_head.state_dict(),
            "best_epoch": best_epoch,
            "test_metrics": test_stats,
        }, path)

        print(
            "Saved pretrained beliefs:",
            path,
            flush=True,
        )

        print(
            "Held-out test:",
            test_stats,
            flush=True,
        )

    finally:
        writer.close()


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--experiment",
        choices=["experiment1", "experiment2"],
        required=True,
    )

    p.add_argument("--output", required=True)
    p.add_argument("--train-per-convention", type=int, default=10000)
    p.add_argument("--val-per-convention", type=int, default=1000)
    p.add_argument("--test-per-convention", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--device", default="cpu")

    pretrain(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
