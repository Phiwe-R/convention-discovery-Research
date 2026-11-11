""" To just vizualized one of the trained Agents (Agent 4)

Place this file in scripts/visualise_trained_agent.py and run from the
repository root, for example:

    python -m scripts.visualise_trained_agent --agent 4 --experiment experiment2

   

This script loads EXISTING actor weights. It never trains or changes a model.
The displayed green/orange goals and ground truth at the end are for
 visibility only; NO hidden goal/convention ID is passed to
any learner policy. Observations match the held-out evaluation scripts.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pygame
import torch

from agents.goal_guided_actor import GoalGuidedJointActor
from agents.goal_only_guided_actor import GoalOnlyGuidedActor
from configs.experiments import EXPERIMENTS, make_env
from training.train_mappo import create_models

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "STAY")
AGENT_NAMES = {
    1: "Implicit GRU",
    2: "Joint convention + goal beliefs",
    3: "Convention-only beliefs",
    4: "Goal-only beliefs",
}


def checkpoint_path(agent, experiment, training_seed, budget):
    """Choose exactly the saved checkpoint family used by the report."""
    if agent in (1, 3):
        group = "implicit" if agent == 1 else "explicit"
        path = (Path("results") / "batch_training" / f"steps_{budget}"
                / experiment / group / f"seed_{training_seed}.pt")
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find Agent {agent} checkpoint: {path}")
        return path

    if agent == 2:
        folder = Path("results") / "goal_guided_clean" / experiment
        prefix = "guided_clean"
    else:
        folder = Path("results") / "goal_only" / experiment
        prefix = "goal_only"

    pattern = re.compile(rf"{prefix}_seed_{training_seed}_steps(\d+)\.pt$")
    candidates = []
    for path in folder.glob(f"{prefix}_seed_{training_seed}_steps*.pt"):
        match = pattern.fullmatch(path.name)
        if match:
            actual = int(match.group(1))
            # Final training jobs can end a few steps past the requested budget.
            if 0 <= actual - budget <= 256:
                candidates.append((actual - budget, path))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find Agent {agent} checkpoint for {experiment}, "
            f"seed={training_seed}, budget={budget} in {folder}"
        )
    return min(candidates)[1]


def load_trained_actor(agent, experiment, training_seed, budget):
    """Use existing evaluation checkpoint formats with strict weight loading."""
    device = torch.device("cpu")
    path = checkpoint_path(agent, experiment, training_seed, budget)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    n_conventions = EXPERIMENTS[experiment]["num_conventions"]
    if ckpt.get("experiment") != experiment:
        raise ValueError(f"Checkpoint experiment does not match: {path}")
    if agent in (1, 3):
        if agent == 3 and not any("belief_net.gru." in key for key in ckpt["actor"]):
            raise RuntimeError(
                "Saved Agent 3 checkpoint has MLP weights, but current "
                "agents/explicit_belief_agent.py defines GRUs. Cannot safely "
                "render that checkpoint with the GRU model. Choose Agent 1, "
                "2, or 4 for a verified trained-policy demo; check Agent 3 "
                "source/checkpoint consistency separately."
            )
        expected = "implicit" if agent == 1 else "explicit"
        if ckpt.get("agent") != expected:
            raise ValueError(f"Expected {expected} checkpoint, got {ckpt.get('agent')}")
        if int(ckpt.get("num_conventions", n_conventions)) != n_conventions:
            raise ValueError("Checkpoint convention count does not match experiment")
        actor, _ = create_models(expected, n_conventions, device)
    elif agent == 2:
        if ckpt.get("architecture") != "joint_goal_guided_clean_v1":
            raise ValueError("Not a final joint goal-guided checkpoint")
        actor = GoalGuidedJointActor(
            num_conventions=n_conventions,
            guide_strength=float(ckpt["guide_strength"]),
        ).to(device)
    else:
        if ckpt.get("architecture") != "goal_only_guided_clean_v1":
            raise ValueError("Not a final goal-only guided checkpoint")
        actor = GoalOnlyGuidedActor(
            guide_strength=float(ckpt["guide_strength"]),
        ).to(device)
    actor.load_state_dict(ckpt["actor"], strict=True)
    actor.eval()
    return actor, path


def initial_memory(actor, agent):
    if agent == 1:
        return actor.initial_hidden(1, torch.device("cpu"))
    return actor.initial_memory(1, torch.device("cpu"))


def policy_action(actor, agent, observation, memory, stochastic=False):
    """Only give the learner-visible 5-value observation to the trained policy."""
    obs = torch.as_tensor(
        np.asarray(observation["learner"], dtype=np.float32),
        dtype=torch.float32,
    ).unsqueeze(0)
    with torch.no_grad():
        if agent in (1, 3):
            action, _logp, _entropy, memory = actor.get_action(
                obs, memory, deterministic=not stochastic,
            )
        else:
            action, _logp, _entropy, memory = actor.get_action(
                obs, memory, deterministic=not stochastic, belief_mode="real",
            )
    return int(action.item()), memory


def belief_summary(agent, memory):
    """Predictions only; never passed to the environment as controls."""
    if agent == 2:
        conv_prob, goal_prob = memory[0][0], memory[1][0]
        return (f" | predicted C{conv_prob.argmax().item()}"
                f" ({conv_prob.max().item():.0%})"
                f" | predicted goal {goal_prob.argmax().item()}"
                f" ({goal_prob.max().item():.0%})")
    if agent == 3:
        conv_prob = memory[0][0]
        return (f" | predicted C{conv_prob.argmax().item()}"
                f" ({conv_prob.max().item():.0%})")
    if agent == 4:
        goal_prob = memory[0][0]
        return (f" | predicted goal {goal_prob.argmax().item()}"
                f" ({goal_prob.max().item():.0%})")
    return ""


def window_closed():
    return any(event.type == pygame.QUIT for event in pygame.event.get())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="experiment2")
    parser.add_argument("--seed", type=int, choices=(20, 21, 22), default=20,
                        help="Training seed of saved actor")
    parser.add_argument("--budget", type=int, choices=(100000, 500000), default=500000)
    parser.add_argument("--episodes", type=int, default=3,
                        help="How many new evaluation episodes to display")
    parser.add_argument("--eval-seed", type=int, default=10000,
                        help="First environment evaluation seed (default matches report)")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="Extra pause after each frame, seconds")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of greedy evaluation")
    parser.add_argument("--convention-id", type=int, default=None,
                        help="Optional forced convention for a focused demo")
    parser.add_argument("--learner-goal-id", type=int, default=None)
    parser.add_argument("--partner-goal-id", type=int, default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.delay < 0:
        parser.error("--episodes must be positive and --delay non-negative")
    n_conventions = EXPERIMENTS[args.experiment]["num_conventions"]
    if args.convention_id is not None and not 0 <= args.convention_id < n_conventions:
        parser.error(f"--convention-id must be 0..{n_conventions-1}")
    for name in ("learner_goal_id", "partner_goal_id"):
        value = getattr(args, name)
        if value is not None and not 0 <= value < 8:
            parser.error(f"--{name.replace('_','-')} must be 0..7")

    actor, path = load_trained_actor(args.agent, args.experiment, args.seed, args.budget)
    print(f"Loaded Agent {args.agent} ({AGENT_NAMES[args.agent]}) from {path}")
    print("Only learner-visible observations are passed into the trained policy.")
    print("The board reveals hidden goals for the SUPERVISOR, not to the actor.")
    print("Use the terminal for predictions and final episode success/failure.\n")

    env = make_env(args.experiment, for_mappo=False, render_mode="human")
    successes = 0
    shown = 0
    options = {}
    for name in ("convention_id", "learner_goal_id", "partner_goal_id"):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    try:
        for episode in range(args.episodes):
            obs, _infos = env.reset(seed=args.eval_seed + episode, options=options)
            memory = initial_memory(actor, args.agent)
            true_c = env.active_convention
            true_g = env.active_learner_goal_id
            print(f"\nEpisode {episode+1}/{args.episodes} | C{true_c} | learner goal ID {true_g}")
            env.render()
            if window_closed():
                break
            done = truncated = False
            while not (done or truncated):
                action, memory = policy_action(actor, args.agent, obs, memory, args.stochastic)
                obs, rewards, terminations, truncations, infos = env.step({"learner": action})
                done = bool(terminations["learner"])
                truncated = bool(truncations["learner"])
                print(f"  Step {env.current_step:2d}: Learner {ACTIONS[action]:5s} "
                      f"at {tuple(env.learner_pos)} | Partner {tuple(env.partner_pos)}"
                      f"{belief_summary(args.agent, memory)}")
                env.render()
                if window_closed():
                    print("Pygame window closed.")
                    return
                time.sleep(args.delay)
            shown += 1
            successes += int(done)
            print(f"  RESULT: {'SUCCESS' if done else 'TIME LIMIT'} "
                  f"in {env.current_step} steps, "
                  f"true convention=C{true_c}, true goal ID={true_g}")
            time.sleep(1.0)
    finally:
        env.close()
    if shown:
        print(f"\nDisplayed {shown} episodes, successes {successes}/{shown}.")
        print("A few visual episodes are a demonstration, not a substitute for the "
              "1,000-episode evaluation results in the report.")


if __name__ == "__main__":
    main()
