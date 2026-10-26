"""Independent candidate goal lists for a matrix-indexed 5x5 toroidal grid."""

LEARNER_GOALS = ((0, 0), (0, 4), (2, 1), (3, 3), (1, 4), (4, 2), (1, 1), (2, 4))
PARTNER_GOALS = ((0, 2), (0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (4, 0), (4, 4))
# Independently sampled start-to-goal distance, NOT linked to a goal ID.
# Three of eight equally likely distance categories are maximum distance four.
DISTANCE_OPTIONS = (1, 2, 2, 3, 3, 4, 4, 4)


def validate_goal_configurations(grid_size=5):
    if grid_size != 5:
        raise ValueError("This goal library targets a 5x5 grid")
    for name, goals in (("learner", LEARNER_GOALS), ("partner", PARTNER_GOALS)):
        if len(goals) != 8 or len(set(goals)) != 8:
            raise ValueError(f"{name} must have exactly eight unique goals")
        if any(len(p) != 2 or any(not isinstance(v, int) or not 0 <= v < grid_size for v in p) for p in goals):
            raise ValueError(f"{name} has invalid matrix coordinates")
    if set(LEARNER_GOALS) & set(PARTNER_GOALS):
        raise ValueError("Goal sets overlap; independent picks could select the same cell")
    if len(DISTANCE_OPTIONS) != 8 or sum(d == 4 for d in DISTANCE_OPTIONS) < 3:
        raise ValueError("Require eight distance options, at least three maximal")
