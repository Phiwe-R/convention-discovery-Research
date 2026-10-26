"""Audit goal-ID leakage from starts for a selected partner goal and distance.
Run: python -m scripts.audit_goal_shortcuts
"""
from collections import defaultdict
from environments.gridworld import SimpleGridworld


def report():
    env = SimpleGridworld(num_conventions=6)
    try:
        evidence = defaultdict(lambda: [0.0] * 8)
        for lg in range(8):
            starts = env._common_start_pool(lg, 2, 4)
            for ls, ps in starts:
                evidence[(ls, ps)][lg] += 1 / len(starts)
        accuracy = 100 * sum(max(v) for v in evidence.values()) / 8
        print(f"Initial-position goal identification with partner goal 2, distance 4: {accuracy:.1f}%")
        print("Baseline 12.5%; diagnostic only, not agent performance.")
    finally:
        env.close()

if __name__ == "__main__":
    report()
