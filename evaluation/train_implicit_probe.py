
"""
Train a frozen-GRU diagnostic probe for convention identification.

The original implicit GRU and policy are NEVER trained here.

Experiment 1: 3 conventions
Experiment 2: 6 conventions

Run from the project root:

python -m evaluation.train_implicit_probe \
    --experiment experiment1 \
    --checkpoint results/experiment1/implicit/seed_20.pt

PowerShell users: put the command on one line.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.experiments import make_env
from evaluation.evaluate_mappo import load_actor


# -------------------------------------------------------
# 1. Small diagnostic classifier
# -------------------------------------------------------

class ConventionProbe(nn.Module):
    """
    Read the frozen GRU's 64-dimensional hidden state.

    The probe predicts which convention produced
    the observation history.

    This is deliberately a LINEAR probe.
    """

    def __init__(self, hidden_dim, num_conventions):
        super().__init__()

        self.classifier = nn.Linear(
            hidden_dim,
            num_conventions
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states)


# -------------------------------------------------------
# 2. Collect hidden states from the pretrained GRU
# -------------------------------------------------------

@torch.no_grad()
def collect_probe_data(
    actor,
    experiment,
    episodes,
    seed_start,
    device
):
    """
    Run the frozen implicit actor in the environment.

    At every timestep:
      1. Read the local learner observation.
      2. Update the GRU hidden state.
      3. Save that hidden state.
      4. Execute the GRU's chosen action.

    The true convention is saved ONLY as a probe label.

    Each episode is reset independently.
    """

    env = make_env(experiment)

    all_hidden_states = []
    all_labels = []
    all_timesteps = []

    successful_episodes = 0

    try:
        for episode_idx in range(episodes):

            episode_seed = seed_start + episode_idx

            observation, info = env.reset(
                seed=episode_seed
            )

            true_convention = int(
                observation["convention"][0]
            )

            # Reset GRU memory for a NEW episode.
            hidden = actor.initial_hidden(
                batch_size=1,
                device=device
            )

            done = False
            truncated = False
            timestep = 0

            while not (done or truncated):

                # Only the learner's local observation
                # is passed into the implicit actor.
                local_obs = np.asarray(
                    observation["learner"],
                    dtype=np.float32
                )

                obs_tensor = torch.as_tensor(
                    local_obs,
                    device=device
                ).unsqueeze(0)

                # Frozen GRU updates its hidden state.
                action, _, _, hidden = actor.get_action(
                    obs_tensor,
                    hidden,
                    deterministic=True
                )

                timestep += 1

                # Shape [1, 1, 64] -> [64].
                hidden_vector = (
                    hidden[0, 0, :]
                    .detach()
                    .cpu()
                    .clone()
                )

                all_hidden_states.append(hidden_vector)

                # This is a training/evaluation label.
                # It never enters the actor.
                all_labels.append(true_convention)

                all_timesteps.append(timestep)

                observation, reward, done, truncated, info = (
                    env.step(int(action.item()))
                )

                if timestep > 10000:
                    raise RuntimeError(
                        "Episode did not terminate"
                    )

            successful_episodes += int(
                bool(info.get("task_success", done))
            )

    finally:
        env.close()

    X = torch.stack(all_hidden_states)

    y = torch.tensor(
        all_labels,
        dtype=torch.long
    )

    t = torch.tensor(
        all_timesteps,
        dtype=torch.long
    )

    print(
        f"Collected {episodes} episodes | "
        f"{len(y)} hidden states | "
        f"Goal success: "
        f"{100 * successful_episodes / episodes:.2f}%"
    )

    return X, y, t


# -------------------------------------------------------
# 3. Evaluate the probe
# -------------------------------------------------------

@torch.no_grad()
def evaluate_probe(
    probe,
    X,
    y,
    timesteps,
    device
):
    """
    Return:
      - Overall timestep identification accuracy
      - Identification accuracy at each timestep
      - Final identification accuracy per episode
      - Cross-entropy loss

    The final timestep of each episode is identified
    by timestep resetting to 1 in the next episode.
    """

    probe.eval()

    logits = probe(X.to(device))

    predictions = logits.argmax(dim=-1).cpu()

    correct = (predictions == y)

    overall_accuracy = (
        100 * correct.float().mean().item()
    )

    loss = F.cross_entropy(
        logits,
        y.to(device)
    ).item()

    accuracy_by_step = {}

    for step in sorted(timesteps.unique().tolist()):

        mask = timesteps == step

        accuracy_by_step[step] = (
            100 * correct[mask].float().mean().item()
        )

    # Find the last recorded timestep of each episode.
    #
    # A new episode starts with timestep == 1.
    final_indices = []

    for i in range(len(timesteps)):

        is_last_record = i == len(timesteps) - 1

        if not is_last_record:
            next_episode_starts = (
                timesteps[i + 1].item() == 1
            )
        else:
            next_episode_starts = False

        if is_last_record or next_episode_starts:
            final_indices.append(i)

    final_correct = correct[final_indices]

    final_accuracy = (
        100 * final_correct.float().mean().item()
    )

    return {
        "overall_accuracy": overall_accuracy,
        "final_accuracy": final_accuracy,
        "accuracy_by_step": accuracy_by_step,
        "cross_entropy": loss
    }


# -------------------------------------------------------
# 4. Train only the probe
# -------------------------------------------------------

def train_probe(
    probe,
    train_X,
    train_y,
    val_X,
    val_y,
    val_t,
    device,
    epochs=100,
    learning_rate=0.001
):
    """
    Train the small classifier.

    The GRU actor is frozen and is NOT included
    in the optimizer.

    The best probe is selected using validation loss.
    """

    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=learning_rate
    )

    train_X = train_X.to(device)
    train_y = train_y.to(device)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):

        probe.train()

        logits = probe(train_X)

        loss = F.cross_entropy(
            logits,
            train_y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Validation evaluation does not train.
        val_metrics = evaluate_probe(
            probe,
            val_X,
            val_y,
            val_t,
            device
        )

        if val_metrics["cross_entropy"] < best_val_loss:

            best_val_loss = val_metrics["cross_entropy"]

            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in probe.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:>3} | "
                f"Train CE {loss.item():.4f} | "
                f"Val CE {val_metrics['cross_entropy']:.4f} | "
                f"Val final ID "
                f"{val_metrics['final_accuracy']:.2f}%"
            )

    # Restore the best validation checkpoint.
    probe.load_state_dict(best_state)

    return probe


# -------------------------------------------------------
# 5. Save identification results
# -------------------------------------------------------

def save_accuracy_csv(path, metrics):
    """
    Save timestep-specific identification accuracy.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

        for step, accuracy in (
            metrics["accuracy_by_step"].items()
        ):
            writer.writerow([step, accuracy])


# -------------------------------------------------------
# 6. Main experiment
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["experiment1", "experiment2"],
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--train-episodes",
        type=int,
        default=3000
    )

    parser.add_argument(
        "--val-episodes",
        type=int,
        default=500
    )

    parser.add_argument(
        "--test-episodes",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--probe-lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    args = parser.parse_args()

    if min(
        args.train_episodes,
        args.val_episodes,
        args.test_episodes,
        args.epochs
    ) <= 0:
        parser.error(
            "Episode counts and epochs must be positive."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    num_conventions = {
        "experiment1": 3,
        "experiment2": 6
    }[args.experiment]

    print("\n=== LOADING FROZEN IMPLICIT GRU ===")

    actor = load_actor(
        args.checkpoint,
        args.experiment,
        "implicit",
        device
    )

    actor.eval()

    # Prevent changes to ALL actor weights.
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    # Record original weights for safety checking.
    original_weights = {
        name: tensor.detach().cpu().clone()
        for name, tensor in actor.state_dict().items()
    }

    print("GRU frozen: YES")

    # Training, validation, and test episode seeds
    # are completely separate.
    #
    # Test starts at 10000 to match the existing
    # evaluation script's default seed_start.
    #
    # Training:   20000, 20001, ...
    # Validation: 30000, 30001, ...
    # Testing:    10000, 10001, ...

    train_start = 20000
    val_start = 30000
    test_start = 10000

    train_seeds = set(
        range(train_start, train_start + args.train_episodes)
    )
    val_seeds = set(
        range(val_start, val_start + args.val_episodes)
    )
    test_seeds = set(
        range(test_start, test_start + args.test_episodes)
    )

    if (
        train_seeds & val_seeds
        or train_seeds & test_seeds
        or val_seeds & test_seeds
    ):
        raise ValueError(
            "Probe train/validation/test seeds overlap."
        )

    print("\n=== COLLECTING TRAINING DATA ===")

    train_X, train_y, train_t = collect_probe_data(
        actor,
        args.experiment,
        args.train_episodes,
        train_start,
        device
    )

    print("\n=== COLLECTING VALIDATION DATA ===")

    val_X, val_y, val_t = collect_probe_data(
        actor,
        args.experiment,
        args.val_episodes,
        val_start,
        device
    )

    print("\n=== COLLECTING TEST DATA ===")

    test_X, test_y, test_t = collect_probe_data(
        actor,
        args.experiment,
        args.test_episodes,
        test_start,
        device
    )

    hidden_dim = train_X.shape[1]

    print(f"\nHidden dimension: {hidden_dim}")

    probe = ConventionProbe(
        hidden_dim=hidden_dim,
        num_conventions=num_conventions
    ).to(device)

    print("\n=== TRAINING LINEAR PROBE ===")

    probe = train_probe(
        probe,
        train_X,
        train_y,
        val_X,
        val_y,
        val_t,
        device,
        epochs=args.epochs,
        learning_rate=args.probe_lr
    )

    print("\n=== FINAL HELD-OUT TEST ===")

    metrics = evaluate_probe(
        probe,
        test_X,
        test_y,
        test_t,
        device
    )

    chance = 100 / num_conventions

    print(f"Chance accuracy: {chance:.2f}%")

    print(
        f"Overall timestep ID accuracy: "
        f"{metrics['overall_accuracy']:.2f}%"
    )

    print(
        f"Final episode ID accuracy: "
        f"{metrics['final_accuracy']:.2f}%"
    )

    print(
        f"Test cross-entropy: "
        f"{metrics['cross_entropy']:.4f}"
    )

    print("\nIdentification accuracy by step (%):")

    for step, accuracy in (
        metrics["accuracy_by_step"].items()
    ):
        print(
            f"Step {step:>2}: {accuracy:.2f}%"
        )

    # Ensure the original actor really stayed frozen.
    for name, original in original_weights.items():

        current = actor.state_dict()[name].detach().cpu()

        if not torch.equal(original, current):
            raise RuntimeError(
                f"Frozen GRU parameter changed: {name}"
            )

    output_dir = (
        Path("results")
        / args.experiment
        / "implicit"
        / f"probe_seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save only the diagnostic classifier.
    torch.save(
        {
            "probe": probe.state_dict(),
            "experiment": args.experiment,
            "num_conventions": num_conventions,
            "hidden_dim": hidden_dim,
            "implicit_checkpoint": args.checkpoint,
            "train_episodes": args.train_episodes,
            "validation_episodes": args.val_episodes,
            "test_episodes": args.test_episodes,
            "test_final_accuracy": metrics["final_accuracy"],
            "test_overall_accuracy": metrics["overall_accuracy"]
        },
        output_dir / "probe.pt"
    )

    save_accuracy_csv(
        output_dir / "identification_by_step.csv",
        metrics
    )

    print("\nFrozen GRU unchanged: YES")
    print(f"Probe saved to: {output_dir / 'probe.pt'}")
    print(
        "Accuracy CSV saved to: "
        f"{output_dir / 'identification_by_step.csv'}"
    )


if __name__ == "__main__":
    main()
