"""Audit whether hidden goal configurations make conventions observationally different.



This is an audit, not agent training. It compares the five-element learner observation
histories under a fixed learner policy (STAY), WITHOUT exposing goal/convention labels
to the observation key. When an episode ends, an END marker is appended thereafter.
"""

import argparse
from collections import Counter, defaultdict

from configs.experiments import make_env
from environments.goal_configurations import LEARNER_GOALS, PARTNER_GOALS


def history_key(prefix):
    """Immutable prefix to group genuinely identical visible histories."""
    return tuple(prefix)


def collect(experiment, seeds, target_distance, horizon):
    n_conventions = 3 if experiment == "experiment1" else 6
    records = []
    for learner_id in range(len(LEARNER_GOALS)):
        for partner_id in range(len(PARTNER_GOALS)):
            for seed in seeds:
                for convention_id in range(n_conventions):
                    env = make_env(experiment, debug=False)
                    try:
                        obs, _ = env.reset(seed=seed, options={
                            "learner_goal_id": learner_id,
                            "partner_goal_id": partner_id,
                            "convention_id": convention_id,
                            "target_learner_distance": target_distance,
                        })
                        # Only the learner's five visible features; NOT obs['convention'].
                        learner = obs["learner"]
                        if tuple(learner.shape) != (5,):
                            raise AssertionError(f"Expected learner observation (5,), got {learner.shape}")
                        hist = [tuple(int(x) for x in learner)]
                        ended = False
                        for _ in range(horizon):
                            if ended:
                                hist.append(("END",))
                                continue
                            obs, _, terminated, truncated, _ = env.step(4)  # STAY
                            hist.append(tuple(int(x) for x in obs["learner"]))
                            ended = bool(terminated or truncated)
                        records.append((convention_id, tuple(hist), (learner_id, partner_id, seed)))
                    finally:
                        env.close()
    return records, n_conventions


def audit(records, n_conventions, horizon):
    """Empirical Bayes-optimal classification ceiling from identical histories.

    For each history we choose its most common convention label. If one history
    maps to different labels, no deterministic observation-only classifier can
    be correct on all those sampled cases.
    """
    print(f"\nScenarios: {len(records)}; conventions: {n_conventions}")
    print("Learner policy: STAY | grouping uses only learner-observable histories")
    print("NOTE: Accuracies are IN-SAMPLE ceilings on these sampled histories, not learned-agent scores.")
    print("Step | ambiguous records | groups mixing convention IDs | best possible accuracy")
    checkpoints = sorted(set((0, 1, 2, 3, 4, 6, 8, horizon)))
    for t in checkpoints:
        if t > horizon:
            continue
        groups = defaultdict(Counter)
        for tau, hist, _ in records:
            groups[history_key(hist[:t + 1])][tau] += 1
        mixed = [c for c in groups.values() if len(c) > 1]
        ambiguous_records = sum(sum(c.values()) for c in mixed)
        best_correct = sum(max(c.values()) for c in groups.values())
        print(f"{t:>4} | {ambiguous_records / len(records):>16.1%} | "
              f"{len(mixed):>28} | {best_correct / len(records):>21.1%}")

    at_end = defaultdict(list)
    for tau, hist, ids in records:
        at_end[hist].append((tau, ids))
    collisions = [items for items in at_end.values() if len({tau for tau, _ in items}) > 1]
    print(f"\nFull-horizon histories shared by DIFFERENT conventions: {len(collisions)}")
    if collisions:
        largest = sorted(collisions, key=len, reverse=True)[:3]
        for i, examples in enumerate(largest, 1):
            print(f"  Example {i}: {len(examples)} scenarios, conventions "
                  f"{sorted({tau for tau, _ in examples})}, "
                  f"first few (convention, learner_goal_id, partner_goal_id, seed): "
                  f"{[(tau, *ids) for tau, ids in examples[:5]]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("experiment1", "experiment2", "both"), default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13])
    parser.add_argument("--distance", type=int, choices=(1, 2, 3, 4), default=4,
                        help="Fixed learner start-to-goal toroidal distance for audit (default: 4)")
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    experiments = ("experiment1", "experiment2") if args.experiment == "both" else (args.experiment,)
    for experiment in experiments:
        print(f"\n=== Hidden-goal ambiguity audit: {experiment} ===")
        records, n_conventions = collect(experiment, args.seeds, args.distance, args.horizon)
        audit(records, n_conventions, args.horizon)
    print("\nInterpretation: If ambiguity remains, the partner convention cannot always be "
          "determined from these observation histories under this action policy. "
          "If ambiguity is zero, that does not prove identification is possible "
          "under every learner policy, start state, or random goal draw.")


if __name__ == "__main__":
    main()
