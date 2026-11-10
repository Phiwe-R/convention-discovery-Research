
"""Train clean explicit goal-guided PPO: Exp1/Exp2, seeds 20-22, 100k/500k."""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def run(command, log_path):
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nRUN:",
        " ".join(map(str, command)),
        flush=True,
    )

    with log_path.open(
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

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        code = process.wait()

    if code:
        raise subprocess.CalledProcessError(
            code,
            command,
        )


def existing_clean_checkpoint(
    experiment,
    seed,
    budget,
    tolerance=256,
):
    root = (
        Path("results/goal_guided_clean")
        / experiment
    )

    pattern = re.compile(
        rf"guided_clean_seed_{seed}_steps(\d+)\.pt$"
    )

    matches = []

    for candidate in root.glob(
        f"guided_clean_seed_{seed}_steps*.pt"
    ):
        match = pattern.fullmatch(
            candidate.name
        )

        if (
            match
            and abs(int(match.group(1)) - budget) <= tolerance
        ):
            matches.append((
                abs(int(match.group(1)) - budget),
                candidate,
            ))

    return min(matches)[1] if matches else None


def rebuild_master():
    root = Path(
        "results/goal_guided_clean/training_logs"
    )

    csv_files = sorted(
        root.glob(
            "experiment*/seed*_target*/training_metrics.csv"
        )
    )

    rows = []

    for path in csv_files:
        with path.open(
            newline="",
            encoding="utf-8",
        ) as fh:
            rows.extend(
                csv.DictReader(fh)
            )

    if not rows:
        return

    rows.sort(
        key=lambda r: (
            r["experiment"],
            int(r["target_timesteps"]),
            int(r["seed"]),
            int(r["step"]),
        )
    )

    dest = Path(
        "results/goal_guided_clean/master_training_curves.csv"
    )

    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with dest.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Master CSV: {dest} ({len(rows)} rows)",
        flush=True,
    )


def ensure_pretrained(experiment, args):
    suffix = (
        "exp1" if experiment == "experiment1"
        else "exp2"
    )

    ckpt = Path(
        f"results/joint_belief/balanced_{suffix}.pt"
    )

    if ckpt.exists():
        print(
            "REUSE belief tracker:",
            ckpt,
            flush=True,
        )
        return ckpt

    print(
        "Missing",
        ckpt,
        "— pretraining before PPO",
        flush=True,
    )

    command = [
        sys.executable,
        "-m",
        "training.pretrain_balanced_belief",

        "--experiment",
        experiment,

        "--output",
        str(ckpt),

        "--train-per-convention",
        str(args.train_per_convention),

        "--val-per-convention",
        str(args.val_per_convention),

        "--test-per-convention",
        str(args.test_per_convention),

        "--epochs",
        str(args.pretrain_epochs),

        "--device",
        args.device,
    ]

    log_path = (
        Path("results/joint_belief/console_logs")
        / f"pretrain_{experiment}.log"
    )

    run(command, log_path)

    if not ckpt.exists():
        raise RuntimeError(
            "Pretraining finished without checkpoint: "
            + str(ckpt)
        )

    return ckpt


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "experiment1",
            "experiment2",
        ],
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

    p.add_argument(
        "--train-per-convention",
        type=int,
        default=10000,
    )

    p.add_argument(
        "--val-per-convention",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--test-per-convention",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--pretrain-epochs",
        type=int,
        default=35,
    )

    p.add_argument(
        "--guide-strength",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

    args = p.parse_args()

    # Pretrain both belief trackers before PPO.
    beliefs = {
        exp: ensure_pretrained(exp, args)
        for exp in args.experiments
    }

    print(
        "\nCLEAN PPO — imitation disabled; "
        "old guided_*.pt NOT eligible for skipping",
        flush=True,
    )

    for exp in args.experiments:
        for budget in args.budgets:
            for seed in args.seeds:

                existing = existing_clean_checkpoint(
                    exp,
                    seed,
                    budget,
                )

                run_name = (
                    f"seed{seed}_target{budget}"
                )

                csv_path = (
                    Path(
                        "results/goal_guided_clean/training_logs"
                    )
                    / exp
                    / run_name
                    / "training_metrics.csv"
                )

                if existing:
                    print(
                        "SKIP:",
                        exp,
                        seed,
                        budget,
                        "->",
                        existing,
                        flush=True,
                    )

                    if not csv_path.exists():
                        print(
                            "WARNING: No historical CSV for this "
                            "checkpoint; cannot reconstruct curves."
                        )

                    rebuild_master()
                    continue

                command = [
                    sys.executable,
                    "-m",
                    "training.train_goal_guided",

                    "--experiment",
                    exp,

                    "--timesteps",
                    str(budget),

                    "--seed",
                    str(seed),

                    "--pretrained-belief",
                    str(beliefs[exp]),

                    "--guide-strength",
                    str(args.guide_strength),

                    "--device",
                    args.device,
                ]

                log_path = (
                    Path("results/goal_guided_clean")
                    / exp
                    / "console_logs"
                    / f"{run_name}.log"
                )

                run(command, log_path)

                # Update combined CSV after each completed run.
                rebuild_master()

    rebuild_master()

    print(
        "\nALL REQUESTED CLEAN TRAINING RUNS COMPLETE. "
        "No evaluation performed."
    )

    print(
        "TensorBoard: "
        "tensorboard --logdir runs/goal_guided_clean"
    )


if __name__ == "__main__":
    main()
