"""Smoke-test both experiments with independently selected goals."""
from configs.experiments import make_env


def main():
    total = 0
    for experiment, count in (("experiment1", 3), ("experiment2", 6)):
        env = make_env(experiment)
        try:
            for lg in range(8):
                for pg in range(8):
                    obs, info = env.reset(seed=42, options={"learner_goal_id": lg, "partner_goal_id": pg})
                    assert info["learner_goal_id"] == lg and info["partner_goal_id"] == pg
                    assert len(obs["learner"]) == 5
                    for c in range(count):
                        obs, info = env.reset(seed=42, options={"learner_goal_id": lg, "partner_goal_id": pg, "convention_id": c})
                        assert info["convention"] == c
                        assert env.observation_space.contains(obs)
                        total += 1
            print(f"{experiment}: all 64 combinations and {count} conventions OK")
        finally:
            env.close()
    print(f"PASS: {total} reset checks")


if __name__ == "__main__":
    main()
