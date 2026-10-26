
"""
Automate MAPPO training across:
- Experiment 1 and Experiment 2
- Implicit and Explicit agents
- 100k, 500k, and 1M timesteps
- Multiple random seeds

Each training run starts from scratch.

Completed checkpoints are skipped automatically.
Each run saves its own checkpoint and training log.

Run from the project root:

python -m scripts.run_all_trainings --dry-run

python -m scripts.run_all_trainings --seeds 20 21 22
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


# --------------------------------------------------
# 1. Default experiment settings
# --------------------------------------------------

DEFAULT_EXPERIMENTS = [
    "experiment1",
    "experiment2",
]

DEFAULT_AGENTS = [
    "implicit",
    "explicit",
]

DEFAULT_TIMESTEPS = [
    100_000,
    500_000
]

DEFAULT_SEEDS = [20, 21, 22]


# --------------------------------------------------
# 2. Check whether a run is already complete
# --------------------------------------------------

def checkpoint_is_complete(path, experiment, agent, seed, timesteps):
    """
    Check that the checkpoint exists and belongs
    to the requested training configuration.

    Training may exceed the requested timestep
    count slightly to finish a complete episode.
    """

    if not path.exists():
        return False

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    return (
        checkpoint.get("experiment") == experiment
        and checkpoint.get("agent") == agent
        and checkpoint.get("seed") == seed
        and checkpoint.get("timesteps", 0) >= timesteps
    )


# --------------------------------------------------
# 3. Save a record of each training run
# --------------------------------------------------

def save_run_record(csv_path, record):
    """
    Append a training result to the batch summary.
    """

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "experiment",
        "agent",
        "seed",
        "requested_timesteps",
        "status",
        "checkpoint",
        "log_file",
        "finished_at",
    ]

    file_exists = csv_path.exists()

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)


# --------------------------------------------------
# 4. Train one model
# --------------------------------------------------

def run_training(
    experiment,
    agent,
    timesteps,
    seed,
    output_root,
    dry_run=False,
    force=False,
):
    """
    Train one model using the existing train_mappo.py.

    Output directories depend on the training budget,
    so different budgets cannot overwrite each other.
    """

    budget_name = f"steps_{timesteps}"

    # Example:
    # results/batch_training/steps_100000/
    training_root = output_root / budget_name

    checkpoint = (
        training_root
        / experiment
        / agent
        / f"seed_{seed}.pt"
    )

    log_file = (
        training_root
        / "logs"
        / experiment
        / agent
        / f"seed_{seed}.log"
    )

    command = [
        sys.executable,
        "-m",
        "training.train_mappo",
        "--experiment",
        experiment,
        "--agent",
        agent,
        "--timesteps",
        str(timesteps),
        "--seed",
        str(seed),
        "--output-dir",
        str(training_root),
    ]

    print("\n" + "=" * 65)
    print("EXPERIMENT:", experiment)
    print("AGENT:", agent)
    print("SEED:", seed)
    print("TIMESTEPS:", f"{timesteps:,}")
    print("CHECKPOINT:", checkpoint)
    print("=" * 65, flush=True)

    if checkpoint.exists() and not force:

        try:
            if checkpoint_is_complete(
                checkpoint,
                experiment,
                agent,
                seed,
                timesteps,
            ):
                print(
                    "SKIPPED: Valid checkpoint already exists.",
                    flush=True,
                )

                return "skipped", checkpoint, log_file

        except Exception as error:
            print(
                f"Existing checkpoint could not be verified: {error}",
                flush=True,
            )

            # Do not silently overwrite a questionable file.
            raise RuntimeError(
                f"Checkpoint exists but is not verified: "
                f"{checkpoint}. Check it or use --force."
            ) from error

        raise RuntimeError(
            f"Checkpoint exists but is incomplete or "
            f"belongs to another configuration: {checkpoint}. "
            f"Check it or use --force."
        )

    print("COMMAND:")
    print(" ".join(command), flush=True)

    if dry_run:
        print("DRY RUN: Training not started.")
        return "dry_run", checkpoint, log_file

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Run one training process at a time.
    # Write its output to both terminal and log.
    with log_file.open(
        "w",
        encoding="utf-8",
    ) as log:

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

            return_code = process.wait()

        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise

    if return_code != 0:
        raise RuntimeError(
            f"Training failed with exit code "
            f"{return_code}. See {log_file}"
        )

    # Verify that training actually saved the right model.
    if not checkpoint_is_complete(
        checkpoint,
        experiment,
        agent,
        seed,
        timesteps,
    ):
        raise RuntimeError(
            f"Training ended but the expected checkpoint "
            f"was not saved correctly: {checkpoint}"
        )

    print("\nTRAINING COMPLETED SUCCESSFULLY")
    print("Saved:", checkpoint, flush=True)

    return "completed", checkpoint, log_file


# --------------------------------------------------
# 5. Run all combinations
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Automate all MAPPO training runs"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=DEFAULT_TIMESTEPS,
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=DEFAULT_EXPERIMENTS,
        default=DEFAULT_EXPERIMENTS,
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        choices=DEFAULT_AGENTS,
        default=DEFAULT_AGENTS,
    )

    parser.add_argument(
        "--output-root",
        default="results/batch_training",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned runs without training",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if a checkpoint exists",
    )

    args = parser.parse_args()

    if any(value <= 0 for value in args.timesteps):
        parser.error("Timesteps must be positive.")

    if any(value < 0 for value in args.seeds):
        parser.error("Seeds must be nonnegative.")

    output_root = Path(args.output_root)

    summary_file = output_root / "training_summary.csv"

    jobs = []

    for seed in args.seeds:
        for timesteps in args.timesteps:
            for experiment in args.experiments:
                for agent in args.agents:

                    jobs.append(
                        (
                            experiment,
                            agent,
                            timesteps,
                            seed,
                        )
                    )

    print("\nBATCH TRAINING PLAN")
    print("-------------------")
    print("Seeds:", args.seeds)
    print("Training budgets:", args.timesteps)
    print("Experiments:", args.experiments)
    print("Agents:", args.agents)
    print("Total training runs:", len(jobs))
    print(
        "Total requested environment steps:",
        f"{sum(job[2] for job in jobs):,}",
    )

    if args.dry_run:
        print("\nDRY RUN ONLY — no training will occur.")

    for index, job in enumerate(jobs, start=1):

        experiment, agent, timesteps, seed = job

        print(
            f"\nRUN {index}/{len(jobs)}",
            flush=True,
        )

        status, checkpoint, log_file = run_training(
            experiment=experiment,
            agent=agent,
            timesteps=timesteps,
            seed=seed,
            output_root=output_root,
            dry_run=args.dry_run,
            force=args.force,
        )

        if not args.dry_run:

            save_run_record(
                summary_file,
                {
                    "experiment": experiment,
                    "agent": agent,
                    "seed": seed,
                    "requested_timesteps": timesteps,
                    "status": status,
                    "checkpoint": str(checkpoint),
                    "log_file": str(log_file),
                    "finished_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                },
            )

    print("\n" + "=" * 65)
    print("BATCH FINISHED")
    print("=" * 65)

    if not args.dry_run:
        print("Training summary:", summary_file)

print("BATCH SCRIPT LOADED", flush=True)
if __name__ == "__main__":
    main()
