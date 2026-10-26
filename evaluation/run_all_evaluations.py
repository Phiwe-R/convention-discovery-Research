
"""
Automated evaluation of all saved MAPPO checkpoints.

For every model:
  1. Evaluate navigation on fixed test episodes.
  2. Measure convention identification.
  3. Train a frozen GRU probe for implicit agents.
  4. Test uniform beliefs for explicit agents.
  5. Save individual and combined results.

Training budgets: 100k and 500k
Training seeds: 20, 21, 22

Run from the project root:

python -m evaluation.run_all_evaluations --dry-run

python -m evaluation.run_all_evaluations --seeds 20 --timesteps 100000

python -m evaluation.run_all_evaluations

Completed evaluations are skipped automatically.
No MAPPO actor or critic weights are updated.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from configs.experiments import make_env

from evaluation.evaluate_mappo import (
    load_actor,
    evaluate,
    save_records,
)

from evaluation.train_implicit_probe import (
    ConventionProbe,
    collect_probe_data,
    train_probe,
    evaluate_probe,
    save_accuracy_csv,
)


DEFAULT_SEEDS = [20, 21, 22]
DEFAULT_TIMESTEPS = [100000, 500000]
EXPERIMENTS = ["experiment1", "experiment2"]
AGENTS = ["implicit", "explicit"]


# ---------------------------------------------------------
# 1. Check that the checkpoint belongs to this experiment
# ---------------------------------------------------------

def verify_checkpoint(path, experiment, agent, seed, steps):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("experiment") != experiment:
        raise ValueError("Checkpoint experiment mismatch")

    if checkpoint.get("agent") != agent:
        raise ValueError("Checkpoint agent mismatch")

    if checkpoint.get("seed") != seed:
        raise ValueError("Checkpoint seed mismatch")

    if checkpoint.get("timesteps", 0) < steps:
        raise ValueError("Checkpoint training incomplete")

    return checkpoint


# ---------------------------------------------------------
# 2. Save identification accuracy at each timestep
# ---------------------------------------------------------

def save_step_accuracy(path, accuracy_by_step):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestep",
            "identification_accuracy_pct"
        ])

        for timestep, accuracy in accuracy_by_step.items():

            writer.writerow([
                timestep,
                accuracy
            ])


# ---------------------------------------------------------
# 3. Evaluate the frozen implicit GRU probe
# ---------------------------------------------------------

def run_implicit_probe(
    actor,
    experiment,
    output_dir,
    seed,
    device,
    train_episodes,
    val_episodes,
    test_episodes,
    probe_epochs,
):

    print("\n=== FROZEN GRU IDENTIFICATION PROBE ===")

    # Freeze the original implicit actor.
    actor.eval()

    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    original_weights = {
        name: value.detach().cpu().clone()
        for name, value in actor.state_dict().items()
    }

    # Separate episode ranges:
    #
    # Probe training:   20000 onward
    # Probe validation: 30000 onward
    # Probe testing:    10000 onward
    #
    # The probe test scenarios match the navigation
    # evaluation scenarios when test_episodes=episodes.

    train_start = 20000
    val_start = 30000
    test_start = 10000

    train_seeds = set(range(
        train_start,
        train_start + train_episodes
    ))

    val_seeds = set(range(
        val_start,
        val_start + val_episodes
    ))

    test_seeds = set(range(
        test_start,
        test_start + test_episodes
    ))

    if (
        train_seeds & val_seeds
        or train_seeds & test_seeds
        or val_seeds & test_seeds
    ):
        raise ValueError(
            "Probe training, validation and test seeds overlap"
        )

    print("Collecting probe training data...")

    train_X, train_y, _ = collect_probe_data(
        actor,
        experiment,
        train_episodes,
        train_start,
        device,
    )

    print("Collecting probe validation data...")

    val_X, val_y, val_t = collect_probe_data(
        actor,
        experiment,
        val_episodes,
        val_start,
        device,
    )

    print("Collecting probe test data...")

    test_X, test_y, test_t = collect_probe_data(
        actor,
        experiment,
        test_episodes,
        test_start,
        device,
    )

    num_conventions = {
        "experiment1": 3,
        "experiment2": 6,
    }[experiment]

    hidden_dim = train_X.shape[1]

    # Reproducible probe initialisation.
    torch.manual_seed(1000 + seed)

    probe = ConventionProbe(
        hidden_dim=hidden_dim,
        num_conventions=num_conventions,
    ).to(device)

    print("\nTraining the linear probe...")

    probe = train_probe(
        probe,
        train_X,
        train_y,
        val_X,
        val_y,
        val_t,
        device,
        epochs=probe_epochs,
        learning_rate=0.001,
    )

    print("\nTesting the frozen-GRU probe...")

    metrics = evaluate_probe(
        probe,
        test_X,
        test_y,
        test_t,
        device,
    )

    # Verify that the original GRU is unchanged.
    for name, before in original_weights.items():

        after = actor.state_dict()[name].detach().cpu()

        if not torch.equal(before, after):
            raise RuntimeError(
                f"Frozen actor changed: {name}"
            )

    torch.save(
        {
            "probe": probe.state_dict(),
            "experiment": experiment,
            "num_conventions": num_conventions,
            "hidden_dim": hidden_dim,
            "seed": seed,
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "test_episodes": test_episodes,
        },
        output_dir / "probe.pt",
    )

    save_accuracy_csv(
        output_dir / "identification_by_step.csv",
        metrics,
    )

    print(
        "Implicit final ID accuracy:",
        f"{metrics['final_accuracy']:.2f}%"
    )

    return metrics


# ---------------------------------------------------------
# 4. Evaluate one trained model
# ---------------------------------------------------------

def evaluate_one_model(
    experiment,
    agent,
    timesteps,
    seed,
    checkpoint_path,
    output_dir,
    episodes,
    device,
    probe_train_episodes,
    probe_val_episodes,
    probe_epochs,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_conventions = {
        "experiment1": 3,
        "experiment2": 6,
    }[experiment]

    actor = load_actor(
        checkpoint_path,
        experiment,
        agent,
        device,
    )

    actor.eval()

    # -------------------------------------------------
    # A. Evaluate navigation using the real actor.
    # -------------------------------------------------

    print("\n=== NAVIGATION EVALUATION ===")

    real_summary, real_records = evaluate(
        actor,
        make_env,
        experiment,
        agent,
        episodes=episodes,
        seed_start=10000,
        belief_mode="real",
        deterministic=True,
    )

    save_records(
        output_dir / "evaluation_real.csv",
        real_records,
    )

    print(
        "Goal success:",
        f"{real_summary['goal_success_pct']:.2f}%"
    )

    print(
        "Mean episode reward:",
        f"{real_summary['mean_reward']:.4f}"
    )

    result = {
        "experiment": experiment,
        "agent": agent,
        "timesteps": timesteps,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "evaluation_episodes": episodes,

        "goal_success_pct": float(
            real_summary["goal_success_pct"]
        ),

        "mean_reward": float(
            real_summary["mean_reward"]
        ),

        "mean_length": float(
            real_summary["mean_length"]
        ),

        "truncated_pct": float(
            real_summary["truncated_pct"]
        ),

        "id_accuracy_final_pct": None,
        "id_accuracy_steps_pct": None,
        "id_cross_entropy": None,

        "uniform_goal_success_pct": None,
        "uniform_mean_reward": None,
        "belief_success_difference_pp": None,

        "chance_id_accuracy_pct": (
            100.0 / num_conventions
        ),
    }

    # -------------------------------------------------
    # B. Explicit belief agent
    # -------------------------------------------------

    if agent == "explicit":

        print("\n=== EXPLICIT IDENTIFICATION ===")

        result["id_accuracy_final_pct"] = float(
            real_summary["id_accuracy_final_pct"]
        )

        accuracy_by_step = (
            real_summary["id_accuracy_by_step_pct"]
        )
        # Count all convention predictions made during evaluation.
        total_predictions = sum(
            len(record["id_by_step"])
            for record in real_records
        )

        # Count how many of those predictions were correct.
        correct_predictions = sum(
            sum(record["id_by_step"])
            for record in real_records
        )

        # Calculate overall identification accuracy.
        result["id_accuracy_steps_pct"] = (
            100 * correct_predictions / total_predictions
        )

        save_step_accuracy(
            output_dir / "identification_by_step.csv",
            accuracy_by_step,
        )

        # -------------------------------------------------
        # C. Uniform-belief intervention
        # -------------------------------------------------

        print("\n=== UNIFORM BELIEF EVALUATION ===")

        uniform_summary, uniform_records = evaluate(
            actor,
            make_env,
            experiment,
            agent,
            episodes=episodes,
            seed_start=10000,
            belief_mode="uniform",
            deterministic=True,
        )

        save_records(
            output_dir / "evaluation_uniform.csv",
            uniform_records,
        )

        uniform_success = float(
            uniform_summary["goal_success_pct"]
        )

        result["uniform_goal_success_pct"] = (
            uniform_success
        )

        result["uniform_mean_reward"] = float(
            uniform_summary["mean_reward"]
        )

        result["belief_success_difference_pp"] = (
            result["goal_success_pct"]
            - uniform_success
        )

        print(
            "Final ID accuracy:",
            f"{result['id_accuracy_final_pct']:.2f}%"
        )

        print(
            "Uniform belief goal success:",
            f"{uniform_success:.2f}%"
        )

    # -------------------------------------------------
    # D. Implicit frozen-GRU probe
    # -------------------------------------------------

    else:

        probe_metrics = run_implicit_probe(
            actor=actor,
            experiment=experiment,
            output_dir=output_dir,
            seed=seed,
            device=torch.device(device),
            train_episodes=probe_train_episodes,
            val_episodes=probe_val_episodes,
            test_episodes=episodes,
            probe_epochs=probe_epochs,
        )

        result["id_accuracy_final_pct"] = float(
            probe_metrics["final_accuracy"]
        )

        result["id_accuracy_steps_pct"] = float(
            probe_metrics["overall_accuracy"]
        )

        result["id_cross_entropy"] = float(
            probe_metrics["cross_entropy"]
        )

    # -------------------------------------------------
    # E. Save one machine-readable summary
    # -------------------------------------------------

    with (
        output_dir / "summary.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print("\nEVALUATION SAVED")
    print(output_dir)

    return result


# ---------------------------------------------------------
# 5. Create the master comparison CSV
# ---------------------------------------------------------

def build_master_csv(output_root):

    summaries = sorted(
        output_root.rglob("summary.json")
    )

    rows = []

    for path in summaries:

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            rows.append(json.load(file))

    if not rows:
        print("No completed evaluations found.")
        return

    rows.sort(
        key=lambda r: (
            r["experiment"],
            r["timesteps"],
            r["seed"],
            r["agent"],
        )
    )

    fieldnames = [
        "experiment",
        "agent",
        "timesteps",
        "seed",
        "checkpoint",
        "evaluation_episodes",
        "goal_success_pct",
        "mean_reward",
        "mean_length",
        "truncated_pct",
        "id_accuracy_final_pct",
        "id_accuracy_steps_pct",
        "id_cross_entropy",
        "chance_id_accuracy_pct",
        "uniform_goal_success_pct",
        "uniform_mean_reward",
        "belief_success_difference_pp",
    ]

    destination = (
        output_root / "master_results.csv"
    )

    with destination.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nMASTER RESULTS SAVED")
    print(destination)
    print("Completed model evaluations:", len(rows))


# ---------------------------------------------------------
# 6. Automatically evaluate all checkpoints
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Batch MAPPO evaluation"
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--timesteps",
        nargs="+",
        type=int,
        default=DEFAULT_TIMESTEPS,
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=EXPERIMENTS,
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        choices=AGENTS,
        default=AGENTS,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--probe-train-episodes",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--probe-val-episodes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--input-root",
        default="results/batch_training",
    )

    parser.add_argument(
        "--output-root",
        default="results/batch_evaluation",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if min(
        args.episodes,
        args.probe_train_episodes,
        args.probe_val_episodes,
        args.probe_epochs,
    ) <= 0:
        parser.error(
            "Episode counts and probe epochs must be positive"
        )

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    jobs = []

    for seed in args.seeds:
        for timesteps in args.timesteps:
            for experiment in args.experiments:
                for agent in args.agents:

                    checkpoint_path = (
                        input_root
                        / f"steps_{timesteps}"
                        / experiment
                        / agent
                        / f"seed_{seed}.pt"
                    )

                    output_dir = (
                        output_root
                        / f"steps_{timesteps}"
                        / experiment
                        / agent
                        / f"seed_{seed}"
                    )

                    jobs.append((
                        experiment,
                        agent,
                        timesteps,
                        seed,
                        checkpoint_path,
                        output_dir,
                    ))

    print("\nBATCH EVALUATION PLAN")
    print("---------------------")
    print("Total models:", len(jobs))
    print("Evaluation episodes per model:", args.episodes)
    print("Seeds:", args.seeds)
    print("Timesteps:", args.timesteps)

    completed = 0
    skipped = 0
    missing = 0

    for index, job in enumerate(jobs, start=1):

        (
            experiment,
            agent,
            timesteps,
            seed,
            checkpoint_path,
            output_dir,
        ) = job

        print("\n" + "=" * 65)
        print(f"MODEL {index}/{len(jobs)}")
        print("Experiment:", experiment)
        print("Agent:", agent)
        print("Timesteps:", timesteps)
        print("Seed:", seed)
        print("Checkpoint:", checkpoint_path)
        print("=" * 65, flush=True)

        if not checkpoint_path.exists():

            print("MISSING CHECKPOINT - SKIPPED")
            missing += 1
            continue

        # Confirm checkpoint metadata before evaluating.
        verify_checkpoint(
            checkpoint_path,
            experiment,
            agent,
            seed,
            timesteps,
        )

        summary_path = (
            output_dir / "summary.json"
        )

        if summary_path.exists() and not args.force:

            with summary_path.open(
                "r",
                encoding="utf-8"
            ) as file:
                previous = json.load(file)

            if (
                previous.get("evaluation_episodes")
                != args.episodes
                or previous.get("checkpoint")
                != str(checkpoint_path)
            ):
                raise RuntimeError(
                    "Existing evaluation uses different "
                    "settings. Check its summary or use --force."
                )

            print("SKIPPED: Evaluation already completed.")
            skipped += 1
            continue

        if args.dry_run:

            print("DRY RUN: Evaluation not started.")
            continue

        # Reproducibility for each model evaluation.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        evaluate_one_model(
            experiment=experiment,
            agent=agent,
            timesteps=timesteps,
            seed=seed,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            episodes=args.episodes,
            device=args.device,
            probe_train_episodes=args.probe_train_episodes,
            probe_val_episodes=args.probe_val_episodes,
            probe_epochs=args.probe_epochs,
        )

        completed += 1

    if not args.dry_run:

        build_master_csv(output_root)

    print("\nBATCH EVALUATION FINISHED")
    print("New evaluations:", completed)
    print("Skipped completed:", skipped)
    print("Missing checkpoints:", missing)


if __name__ == "__main__":
    main()
