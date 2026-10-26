"""View partner conventions: python -m scripts.visualise_conventions --experiment experiment2

Requires a desktop display. The learner follows a scripted test strategy,
not a trained agent; this script is strictly for environment demonstration.
"""
import argparse
import time
import numpy as np
from configs.experiments import make_env, EXPERIMENTS


def scripted_action(env):
    learner = env.learner_pos
    partner = env.partner_pos
    index = env.convention_step
    action = env.partner_actions[index] if index < len(env.partner_actions) else 4
    next_partner = env._move(partner, action)
    if not np.array_equal(next_partner, learner):
        return 4
    reserved = None
    if index + 1 < len(env.partner_actions):
        reserved = env._move(next_partner, env.partner_actions[index + 1])
    for candidate_action in (0, 1, 2, 3):
        candidate = env._move(learner, candidate_action)
        if any(np.array_equal(candidate, p) for p in (partner, next_partner, env.learner_goal)):
            continue
        if reserved is not None and np.array_equal(candidate, reserved):
            continue
        return candidate_action
    return 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="experiment1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learner-goal-id", type=int, choices=range(8), default=5)
    parser.add_argument("--partner-goal-id", type=int, choices=range(8), default=2)
    args = parser.parse_args()
    environment = make_env(args.experiment, for_mappo=False, render_mode="human", debug=True)
    try:
        for convention_id in range(EXPERIMENTS[args.experiment]["num_conventions"]):
            print(f"\nTesting convention C{convention_id}, learner goal {args.learner_goal_id}, partner goal {args.partner_goal_id}")
            environment.reset(seed=args.seed, options={"convention_id": convention_id, "learner_goal_id": args.learner_goal_id, "partner_goal_id": args.partner_goal_id})
            environment.render()
            for _ in range(environment.max_steps):
                if not environment.agents:
                    break
                environment.step({"learner": scripted_action(environment)})
                environment.render()
                if environment.convention_step == len(environment.partner_actions):
                    break
            time.sleep(1)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
