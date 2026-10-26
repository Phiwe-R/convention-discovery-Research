
"""
Audit hidden-goal uncertainty and convention-dependent partner behaviour.

Uses the actual gridworld.py implementation.

Checks:
1. How often identical initial observations hide different goals.
2. Whether those goals require different shortest-path actions.
3. How often different conventions produce different partner signals.

No agent training or checkpoint changes.

Run:
python -m scripts.audit_convention_goal_dependence --experiment experiment1
"""

import argparse
from collections import defaultdict

import numpy as np

from configs.experiments import make_env


ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "STAY")


# -------------------------------------------------------
# 1. Find shortest-path actions towards the hidden goal
# -------------------------------------------------------

def shortest_path_actions(world, learner_position, learner_goal):
    """
    Return all actions that reduce toroidal distance
    to the true learner goal by one.

    Multiple directions can be equally good.

    This calculation ignores moving-agent collisions.
    """

    distance_before = world._toroidal_heuristic(
        learner_position,
        learner_goal
    )

    best_actions = set()

    for action in range(4):

        next_position = world._move(
            learner_position,
            action
        )

        distance_after = world._toroidal_heuristic(
            next_position,
            learner_goal
        )

        if distance_after == distance_before - 1:
            best_actions.add(action)

    return best_actions


# -------------------------------------------------------
# 2. Compare conventions in exactly the same world
# -------------------------------------------------------

def get_first_partner_actions(world):
    """
    Keep the SAME:
      - learner start
      - learner goal
      - partner start
      - partner goal

    Change ONLY the partner convention.

    Return the first planned partner action for
    every available convention.
    """

    first_actions = {}

    for convention_id, convention in world.conventions.items():

        policy = convention["policy"]

        action_sequence = world._generate_partner_actions(
            policy
        )

        if len(action_sequence) > 0:
            first_action = int(action_sequence[0])
        else:
            first_action = 4

        first_actions[convention_id] = first_action

    return first_actions


# -------------------------------------------------------
# 3. Sample real environment resets
# -------------------------------------------------------

def collect_samples(experiment, episodes, seed_start):
    """
    Sample episodes using the actual reset() implementation.

    This automatically respects:
      - independent goal sampling
      - distance options
      - valid start-position pools
      - all six convention feasibility constraints

    Group scenarios by what the learner can actually
    observe BEFORE the partner first moves.
    """

    env = make_env(experiment)

    groups = defaultdict(list)

    first_signal_changes = 0

    example_signal = None

    try:

        world = env.env

        for episode_index in range(episodes):

            seed = seed_start + episode_index

            observation, info = env.reset(seed=seed)

            # The learner is allowed to see ONLY this.
            initial_observation = tuple(
                int(value)
                for value in observation["learner"]
            )

            # Hidden quantities are used ONLY for analysis.
            learner_start = tuple(
                int(value)
                for value in world.learner_pos
            )

            learner_goal = tuple(
                int(value)
                for value in world.learner_goal
            )

            partner_goal = tuple(
                int(value)
                for value in world.partner_goal
            )

            convention_id = int(
                world.active_convention
            )

            optimal_actions = shortest_path_actions(
                world,
                learner_start,
                learner_goal
            )

            # Counterfactual convention comparison:
            # all goals and positions are held fixed.
            first_actions = get_first_partner_actions(
                world
            )

            different_signals = (
                len(set(first_actions.values())) > 1
            )

            if different_signals:
                first_signal_changes += 1

                if example_signal is None:
                    example_signal = {
                        "observation": initial_observation,
                        "learner_goal": learner_goal,
                        "partner_goal": partner_goal,
                        "first_actions": first_actions,
                    }

            groups[initial_observation].append({
                "learner_goal": learner_goal,
                "partner_goal": partner_goal,
                "convention": convention_id,
                "optimal_actions": optimal_actions,
            })

    finally:
        env.close()

    return groups, first_signal_changes, example_signal


# -------------------------------------------------------
# 4. Analyse hidden-goal uncertainty
# -------------------------------------------------------

def analyse_groups(groups, episodes):
    """
    Examine scenarios with the same initial observation.

    If different compatible goals have no common
    shortest-path action, hidden-goal uncertainty
    creates conflicting immediate navigation choices.

    These are sample-based estimates, not exhaustive
    probabilities over every possible hidden world.
    """

    ambiguous_samples = 0
    conflicting_samples = 0
    best_action_correct = 0

    examples = []

    for observation, scenarios in groups.items():

        count = len(scenarios)

        possible_goals = {
            scenario["learner_goal"]
            for scenario in scenarios
        }

        if len(possible_goals) > 1:
            ambiguous_samples += count

        # Find actions that work for EVERY sampled
        # hidden goal compatible with this observation.
        common_actions = set(range(4))

        for scenario in scenarios:

            common_actions &= scenario[
                "optimal_actions"
            ]

        if len(common_actions) == 0:

            conflicting_samples += count

            if len(examples) < 3 and len(scenarios) > 1:

                examples.append({
                    "observation": observation,
                    "scenarios": scenarios,
                })

        # Suppose the learner cannot see its hidden goal.
        #
        # Which ONE action works for the most sampled
        # scenarios with this initial observation?

        best_count = 0

        for action in range(4):

            correct_count = sum(
                action in scenario["optimal_actions"]
                for scenario in scenarios
            )

            best_count = max(
                best_count,
                correct_count
            )

        best_action_correct += best_count

    return {
        "distinct_observations": len(groups),

        "ambiguous_goal_pct": (
            100 * ambiguous_samples / episodes
        ),

        "conflicting_actions_pct": (
            100 * conflicting_samples / episodes
        ),

        "best_action_agreement_pct": (
            100 * best_action_correct / episodes
        ),

        "examples": examples,
    }


# -------------------------------------------------------
# 5. Print results
# -------------------------------------------------------

def run_audit(experiment, episodes, seed_start):

    print("\n====================================")
    print(" CONVENTION-GOAL DEPENDENCE AUDIT")
    print("====================================")

    print("Experiment:", experiment)
    print("Sampled episodes:", episodes)

    env = make_env(experiment)

    try:
        num_conventions = len(env.env.conventions)
    finally:
        env.close()

    print("Number of conventions:", num_conventions)
    print("Learner goal candidates: 8")
    print("Partner goal candidates: 8")
    print("Possible goal pairs: 64")

    print("\n=== 1. COLLECTING ENVIRONMENT STATES ===")

    groups, signal_changes, example_signal = (
        collect_samples(
            experiment,
            episodes,
            seed_start
        )
    )

    results = analyse_groups(
        groups,
        episodes
    )

    print("\n=== 2. INITIAL GOAL UNCERTAINTY ===")

    print(
        "Distinct initial observations:",
        results["distinct_observations"]
    )

    print(
        "Samples whose initial observation "
        "has multiple compatible sampled goals:"
    )

    print(
        f"{results['ambiguous_goal_pct']:.2f}%"
    )

    print("\n=== 3. NAVIGATION ACTION CONFLICT ===")

    print(
        "Samples whose initial observation has "
        "no common shortest-path action across "
        "its sampled hidden goals:"
    )

    print(
        f"{results['conflicting_actions_pct']:.2f}%"
    )

    print(
        "\nBest initial shortest-path action agreement "
        "without observing the hidden goal:"
    )

    print(
        f"{results['best_action_agreement_pct']:.2f}%"
    )

    print("\n=== 4. CONVENTION-DEPENDENT SIGNALS ===")

    signal_pct = (
        100 * signal_changes / episodes
    )

    print(
        "Worlds where changing ONLY the convention "
        "changes the first planned partner action:"
    )

    print(f"{signal_pct:.2f}%")

    if example_signal is not None:

        print("\nExample using the same world:")

        print(
            "Initial observation:",
            example_signal["observation"]
        )

        print(
            "Hidden learner goal:",
            example_signal["learner_goal"]
        )

        print(
            "Hidden partner goal:",
            example_signal["partner_goal"]
        )

        for convention_id, action in (
            example_signal["first_actions"].items()
        ):

            print(
                f"C{convention_id}: "
                f"{ACTION_NAMES[action]}"
            )

    print("\n=== 5. CONFLICTING GOAL EXAMPLES ===")

    for index, example in enumerate(
        results["examples"],
        start=1
    ):

        print(f"\nExample {index}")

        print(
            "Same learner observation:",
            example["observation"]
        )

        unique_examples = {}

        for scenario in example["scenarios"]:

            goal = scenario["learner_goal"]

            if goal not in unique_examples:
                unique_examples[goal] = scenario[
                    "optimal_actions"
                ]

        for goal, actions in unique_examples.items():

            names = [
                ACTION_NAMES[action]
                for action in sorted(actions)
            ]

            print(
                f"Hidden goal {goal} "
                f"-> shortest actions {names}"
            )

    print("\n=== 6. CONCLUSION ===")

    print(
        "The convention does not directly determine "
        "the sampled learner goal."
    )

    print(
        "The partner's convention can change the "
        "signals available to interpret that goal."
    )

    print(
        "This audit measures INITIAL uncertainty "
        "and FIRST partner signals only."
    )

    print(
        "It does not yet measure whether knowing "
        "the convention improves action selection "
        "after observing a partner trajectory."
    )

    print(
        "\nNo environment files, model weights, "
        "or checkpoints were modified."
    )


# -------------------------------------------------------
# 6. Command line
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["experiment1", "experiment2"],
        required=True
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=40000
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error(
            "--episodes must be positive"
        )

    run_audit(
        experiment=args.experiment,
        episodes=args.episodes,
        seed_start=args.seed_start
    )


if __name__ == "__main__":
    main()
