"""The SINGLE library of deterministic partner conventions C0--C5.

All plans are generated at the beginning of each episode using its sampled
start positions and hidden learner goal; these are not fixed action lists.

Actions (matrix coordinates): UP=0 (row - 1), DOWN=1 (row + 1),
LEFT=2 (column - 1), RIGHT=3 (column + 1), STAY=4.
Movement and A* use the environment's toroidal 5x5 grid mechanics.
"""

from typing import Callable, Sequence
import numpy as np

# Stable IDs across every experiment. Do not reorder: checkpoints refer to IDs.
CONVENTION_LIBRARY = {
    0: {"name": "goal_visit", "policy": "astar_visit_and_stay"},
    1: {"name": "route_mirroring", "policy": "astar_mirror_learner_route"},
    2: {"name": "inverse_mirroring", "policy": "astar_inverse_mirror_learner_route"},
    3: {"name": "clockwise_route_rotation", "policy": "astar_rotate_clockwise"},
    4: {"name": "counterclockwise_route_rotation", "policy": "astar_rotate_counterclockwise"},
    5: {"name": "offset_goal_waypoint", "policy": "astar_offset_waypoint_and_stay"},
}

CLOCKWISE = {0: 3, 3: 1, 1: 2, 2: 0}
COUNTERCLOCKWISE = {0: 2, 2: 1, 1: 3, 3: 0}
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


def get_conventions(num_conventions: int) -> dict:
    """Choose the first three (Experiment 1) or all six (Experiment 2)."""
    if num_conventions not in (3, 6):
        raise ValueError("num_conventions must be 3 (Experiment 1) or 6 (Experiment 2)")
    return {i: CONVENTION_LIBRARY[i].copy() for i in range(num_conventions)}


def generate_partner_actions(
    policy: str,
    *,
    learner_start: np.ndarray,
    learner_goal: np.ndarray,
    partner_start: np.ndarray,
    partner_goal: np.ndarray,
    astar: Callable,
    move: Callable,
) -> list[int]:
    """Produce one partner trajectory using an episode's state.

    `astar(start, goal, blocked=None)` returns movement action IDs.
    `move(position, action)` implements the shared grid topology.
    Neither the actor nor its observation is passed a convention ID.
    """
    if policy == "astar_visit_and_stay":  # C0
        outward = astar(partner_start, learner_goal)
        returning = astar(learner_goal, partner_goal)
        actions = outward + [4] + returning

    elif policy in (  # C1 and C2
        "astar_mirror_learner_route", "astar_inverse_mirror_learner_route"
    ):
        route = astar(learner_start, learner_goal)
        demonstration = ([OPPOSITE[a] for a in route] if
                         policy == "astar_inverse_mirror_learner_route" else route)
        position = partner_start.copy()
        for action in demonstration:
            position = move(position, action)
            if np.array_equal(position, learner_goal):
                raise ValueError("Mirroring demonstration would visit hidden learner goal")
        returning = astar(position, partner_goal, blocked={tuple(learner_goal)})
        actions = demonstration + returning

    elif policy in (  # C3 and C4
        "astar_rotate_clockwise", "astar_rotate_counterclockwise"
    ):
        route = astar(learner_start, learner_goal)
        rotation = (CLOCKWISE if policy == "astar_rotate_clockwise" else COUNTERCLOCKWISE)
        demonstration = [rotation[a] for a in route]
        position = partner_start.copy()
        for action in demonstration:
            position = move(position, action)
            if np.array_equal(position, learner_goal):
                raise ValueError("Rotated demonstration would visit hidden learner goal")
        returning = astar(position, partner_goal, blocked={tuple(learner_goal)})
        actions = demonstration + returning

    elif policy == "astar_offset_waypoint_and_stay":  # C5
        waypoint = move(learner_goal, 3)  # one toroidal RIGHT from hidden goal
        if np.array_equal(waypoint, partner_goal):
            raise ValueError("Offset waypoint coincides with partner goal")
        outward = astar(partner_start, waypoint, blocked={tuple(learner_goal)})
        returning = astar(waypoint, partner_goal, blocked={tuple(learner_goal)})
        actions = outward + [4] + returning
    else:
        raise ValueError(f"Unknown convention policy: {policy}")

    # Same goal-reachability and non-visiting checks as in the original files.
    position = partner_start.copy()
    visited = []
    for action in actions:
        position = move(position, action)
        visited.append(tuple(map(int, position)))
    if not np.array_equal(position, partner_goal):
        raise RuntimeError("Partner convention did not finish at its own goal")

    if policy == "astar_visit_and_stay":
        if tuple(learner_goal) not in visited or 4 not in actions:
            raise RuntimeError("Goal-visit signal was not generated")
    else:
        if tuple(learner_goal) in visited:
            raise RuntimeError("Non-visiting convention entered learner goal")
    if policy == "astar_offset_waypoint_and_stay":
        if 4 not in actions or tuple(move(learner_goal, 3)) not in visited:
            raise RuntimeError("Offset waypoint signal was not generated")
    return actions
