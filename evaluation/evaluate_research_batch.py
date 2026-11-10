
"""Evaluates  explicit (Agent 2 , both goal and convention) and implicit GRU(Agent 1)  to make a versus
this script outputs final_evaluations folder under results 

final evaluations is for Agent 2 evaluations


I didnt include Implict vs Agent 2 in the report
 
Both experiments, seeds 20/21/22, budgets 100k/500k.
No identification metrics are calculated for implicit GRU.

Run from project root:
    python -m evaluation.evaluate_research_batch
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from agents.goal_guided_actor import GoalGuidedJointActor
from configs.experiments import make_env
from training.train_mappo import create_models


COMMON = [
    "eval_seed",
    "goal_success",
    "reward",
    "episode_length",
    "truncated",
]

MODES = (
    "real",
    "uniform",
    "uniform_goal",
    "uniform_convention",
)


def save_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))



def find_implicit(master_csv, experiment, seed, budget):
    """Locate an already-trained implicit GRU checkpoint."""

    checkpoint = (
        Path("results")
        / "batch_training"
        / f"steps_{budget}"
        / experiment
        / "implicit"
        / f"seed_{seed}.pt"
    )

    if checkpoint.is_file():
        return checkpoint

    raise FileNotFoundError(
        f"Implicit checkpoint not found: {checkpoint}\n"
        "Check that your old implicit training results "
        "are present in results/batch_training/."
    )


def find_explicit(experiment, seed, budget, root, tolerance=256):
    """Find CLEAN explicit model, not old imitation-trained model."""

    folder = Path(root) / experiment

    pattern = re.compile(
        rf"guided_clean_seed_{seed}_steps(\d+)\.pt$"
    )

    candidates = []

    for path in folder.glob(
        f"guided_clean_seed_{seed}_steps*.pt"
    ):
        match = pattern.fullmatch(path.name)

        if match:
            actual_steps = int(match.group(1))

            if abs(actual_steps - budget) <= tolerance:
                candidates.append(
                    (abs(actual_steps - budget), path)
                )

    if not candidates:
        raise FileNotFoundError(
            f"Missing clean explicit checkpoint: "
            f"{experiment}, seed={seed}, budget={budget}. "
            f"Folder: {folder}"
        )

    return min(candidates)[1]


def load_model(path, agent, experiment, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    n = {
        "experiment1": 3,
        "experiment2": 6,
    }[experiment]

    if checkpoint.get("experiment") != experiment:
        raise ValueError(f"Wrong experiment in {path}")

    if int(checkpoint.get("num_conventions", n)) != n:
        raise ValueError(f"Wrong number of conventions in {path}")

    if agent == "implicit":

        if checkpoint.get("agent") not in (None, "implicit"):
            raise ValueError(f"Not an implicit model: {path}")

        actor, _ = create_models(
            "implicit", n, device
        )

    else:

        if checkpoint.get("architecture") != "joint_goal_guided_clean_v1":
            raise ValueError(
                f"Not a clean goal-guided model: {path}"
            )

        actor = GoalGuidedJointActor(
            num_conventions=n,
            guide_strength=float(checkpoint["guide_strength"]),
        ).to(device)

    actor.load_state_dict(
        checkpoint["actor"],
        strict=True,
    )

    actor.eval()

    return actor, int(checkpoint.get("timesteps", 0))


def evaluate(
    actor,
    experiment,
    agent,
    mode,
    episodes,
    seed_start,
    device,
):
    """Evaluate on identical environment reset seeds."""

    env = make_env(experiment)

    navigation = []
    identification = []

    # timestep -> [observations, correct conventions, correct goals]
    by_step = defaultdict(lambda: [0, 0, 0])

    try:
        with torch.no_grad():

            for seed in range(seed_start, seed_start + episodes):

                obs, info = env.reset(seed=seed)

                if agent == "implicit":
                    memory = actor.initial_hidden(1, device)

                else:
                    memory = actor.initial_memory(1, device)

                    true_c = int(obs["convention"][0])
                    true_g = int(info["learner_goal_id"])

                    c_correct = 0
                    g_correct = 0

                finished = False
                time_limit = False

                reward_total = 0.0
                t = 0
                last_info = info

                while not (finished or time_limit):

                    observation = torch.as_tensor(
                        np.asarray(
                            obs["learner"],
                            dtype=np.float32,
                        ),
                        device=device,
                    ).unsqueeze(0)

                    if agent == "implicit":

                        action, _, _, memory = actor.get_action(
                            observation,
                            memory,
                            deterministic=True,
                        )

                    else:

                        action, _, _, memory = actor.get_action(
                            observation,
                            memory,
                            deterministic=True,
                            belief_mode=mode,
                        )

                        # Only the explicit agent is assessed
                        # on convention and goal identification.
                        if mode == "real":

                            predicted_c = int(
                                memory[0].argmax().item()
                            )

                            predicted_g = int(
                                memory[1].argmax().item()
                            )

                            c_correct = int(
                                predicted_c == true_c
                            )

                            g_correct = int(
                                predicted_g == true_g
                            )

                            counts = by_step[t + 1]

                            counts[0] += 1
                            counts[1] += c_correct
                            counts[2] += g_correct

                    obs, reward, finished, time_limit, last_info = (
                        env.step(int(action.item()))
                    )

                    reward_total += float(reward)
                    t += 1

                    if t > 10000:
                        raise RuntimeError(
                            "Environment did not terminate"
                        )

                success = int(
                    bool(
                        last_info.get(
                            "task_success",
                            finished,
                        )
                    )
                )

                navigation.append({
                    "eval_seed": seed,
                    "goal_success": success,
                    "reward": reward_total,
                    "episode_length": t,
                    "truncated": int(bool(time_limit)),
                })

                if agent == "explicit" and mode == "real":

                    identification.append({
                        "eval_seed": seed,
                        "true_convention": true_c,
                        "true_goal_id": true_g,
                        "convention_correct_final": c_correct,
                        "goal_correct_final": g_correct,
                        "episode_length": t,
                    })

    finally:
        env.close()

    timestep_rows = []

    for timestep, counts in sorted(by_step.items()):

        total = counts[0]

        timestep_rows.append({
            "timestep": timestep,
            "observations": total,
            "convention_id_pct": 100 * counts[1] / total,
            "goal_id_pct": 100 * counts[2] / total,
        })

    return navigation, identification, timestep_rows


def valid_cache(path, seed_start, episodes):
    """Skip evaluation only if all requested episode seeds exist."""

    if not Path(path).exists():
        return False

    rows = read_csv(path)

    return (
        len(rows) == episodes
        and int(rows[0]["eval_seed"]) == seed_start
        and int(rows[-1]["eval_seed"])
        == seed_start + episodes - 1
    )


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--experiments",
        nargs="+",
        default=["experiment1", "experiment2"],
        choices=["experiment1", "experiment2"],
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20, 21, 22],
    )

    p.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[100000, 500000],
    )

    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed-start", type=int, default=10000)

    p.add_argument(
        "--master",
        default="master_results.csv",
    )

    p.add_argument(
        "--explicit-root",
        default="results/goal_guided_clean",
    )

    p.add_argument(
        "--output",
        default="results/final_evaluation",
    )

    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")

    args = p.parse_args()

    if args.episodes <= 0:
        p.error("--episodes must be positive")

    if args.device == "cpu":
        torch.set_num_threads(1)

    device = torch.device(args.device)
    out = Path(args.output)

    shared_summaries = []
    id_summaries = []
    id_by_timestep = []

    for experiment in args.experiments:

        for budget in args.budgets:

            for seed in args.seeds:

                for agent in ("implicit", "explicit"):

                    if agent == "implicit":

                        checkpoint = find_implicit(
                            args.master,
                            experiment,
                            seed,
                            budget,
                        )

                        modes = ("real",)

                    else:

                        checkpoint = find_explicit(
                            experiment,
                            seed,
                            budget,
                            args.explicit_root,
                        )

                        modes = MODES

                    model, actual_steps = load_model(
                        checkpoint,
                        agent,
                        experiment,
                        device,
                    )

                    label = (
                        "implicit"
                        if agent == "implicit"
                        else "explicit_goal_guided_clean"
                    )

                    run_dir = (
                        out
                        / "episodes"
                        / experiment
                        / label
                    )

                    for mode in modes:

                        stem = (
                            f"seed{seed}_budget{budget}_{mode}"
                        )

                        nav_path = run_dir / f"{stem}.csv"
                        id_path = (
                            run_dir / f"{stem}_identification.csv"
                        )
                        step_path = (
                            run_dir / f"{stem}_by_timestep.csv"
                        )

                        cache = valid_cache(
                            nav_path,
                            args.seed_start,
                            args.episodes,
                        )

                        if agent == "explicit" and mode == "real":

                            cache = (
                                cache
                                and valid_cache(
                                    id_path,
                                    args.seed_start,
                                    args.episodes,
                                )
                                and step_path.exists()
                            )

                        if cache and not args.force:

                            nav = read_csv(nav_path)

                            if agent == "explicit" and mode == "real":
                                ids = read_csv(id_path)
                                steps = read_csv(step_path)
                            else:
                                ids = []
                                steps = []

                            print(
                                "SKIP (cached)",
                                experiment,
                                budget,
                                seed,
                                label,
                                mode,
                                flush=True,
                            )

                        else:

                            print(
                                "EVAL",
                                experiment,
                                budget,
                                seed,
                                label,
                                mode,
                                flush=True,
                            )

                            nav, ids, steps = evaluate(
                                model,
                                experiment,
                                agent,
                                mode,
                                args.episodes,
                                args.seed_start,
                                device,
                            )

                            save_csv(
                                nav_path,
                                nav,
                                COMMON,
                            )

                            if ids:

                                save_csv(
                                    id_path,
                                    ids,
                                    [
                                        "eval_seed",
                                        "true_convention",
                                        "true_goal_id",
                                        "convention_correct_final",
                                        "goal_correct_final",
                                        "episode_length",
                                    ],
                                )

                                save_csv(
                                    step_path,
                                    steps,
                                    [
                                        "timestep",
                                        "observations",
                                        "convention_id_pct",
                                        "goal_id_pct",
                                    ],
                                )

                        row = {
                            "experiment": experiment,
                            "agent": label,
                            "seed": seed,
                            "target_timesteps": budget,
                            "actual_timesteps": actual_steps,
                            "belief_mode": mode,
                            "episodes": len(nav),
                            "seed_start": args.seed_start,
                            "checkpoint": str(checkpoint),

                            "goal_success_pct": 100 * mean(
                                float(r["goal_success"])
                                for r in nav
                            ),

                            "mean_reward": mean(
                                float(r["reward"])
                                for r in nav
                            ),

                            "mean_length": mean(
                                float(r["episode_length"])
                                for r in nav
                            ),

                            "truncated_pct": 100 * mean(
                                float(r["truncated"])
                                for r in nav
                            ),
                        }

                        shared_summaries.append(row)

                        if ids:

                            id_summaries.append({
                                "experiment": experiment,
                                "seed": seed,
                                "target_timesteps": budget,
                                "episodes": len(ids),

                                "convention_id_final_pct": 100 * mean(
                                    float(r["convention_correct_final"])
                                    for r in ids
                                ),

                                "goal_id_final_pct": 100 * mean(
                                    float(r["goal_correct_final"])
                                    for r in ids
                                ),
                            })

                            for r in steps:

                                id_by_timestep.append({
                                    "experiment": experiment,
                                    "seed": seed,
                                    "target_timesteps": budget,
                                    **r,
                                })

                        # Save progress after each evaluation.
                        save_csv(
                            out / "checkpoint_evaluations.csv",
                            shared_summaries,
                            list(shared_summaries[0]),
                        )

                        if id_summaries:

                            save_csv(
                                out / "explicit_identification.csv",
                                id_summaries,
                                list(id_summaries[0]),
                            )

                            save_csv(
                                out / "explicit_identification_by_timestep.csv",
                                id_by_timestep,
                                list(id_by_timestep[0]),
                            )

                        print(
                            f"  Success {row['goal_success_pct']:.1f}% "
                            f"| Reward {row['mean_reward']:.3f}",
                            flush=True,
                        )

                    del model

    # Shared comparison: only real-belief explicit
    # and original implicit model.
    real = [
        r for r in shared_summaries
        if r["belief_mode"] == "real"
    ]

    save_csv(
        out / "implicit_vs_explicit.csv",
        real,
        list(real[0]),
    )

    # Mean and sample standard deviation across training seeds.
    groups = defaultdict(list)

    for row in real:
        key = (
            row["experiment"],
            row["target_timesteps"],
            row["agent"],
        )
        groups[key].append(row)

    aggregates = []

    for (exp, budget, agent), rows in sorted(groups.items()):

        agg = {
            "experiment": exp,
            "target_timesteps": budget,
            "agent": agent,
            "training_seeds": len(rows),
        }

        for field in (
            "goal_success_pct",
            "mean_reward",
            "mean_length",
            "truncated_pct",
        ):

            values = [
                float(r[field])
                for r in rows
            ]

            agg[f"{field}_mean"] = mean(values)

            agg[f"{field}_sd"] = (
                stdev(values)
                if len(values) > 1
                else 0.0
            )

        aggregates.append(agg)

    save_csv(
        out / "comparison_mean_sd.csv",
        aggregates,
        list(aggregates[0]),
    )

    print("\nEVALUATION COMPLETED")
    print("Results folder:", out)
    print("Comparison: implicit_vs_explicit.csv")
    print("Summary: comparison_mean_sd.csv")
    print("Identification: explicit_identification.csv")
    print("Timestep ID: explicit_identification_by_timestep.csv")
    print("Ablations: checkpoint_evaluations.csv")


if __name__ == "__main__":
    main()
