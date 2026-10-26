"""Central experiment configuration -- no duplicated GridWorld files."""
from environments.gridworld import SimpleGridworld, MAPPOWrapper

EXPERIMENTS = {
    "experiment1": {"num_conventions": 3},  # C0, C1, C2
    "experiment2": {"num_conventions": 6},  # C0, C1, C2, C3, C4, C5
}


def make_env(experiment: str, *, for_mappo: bool = True, **kwargs):
    """Instantiate the exact same GridWorld class for both experiments."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; choose {list(EXPERIMENTS)}")
    constructor = MAPPOWrapper if for_mappo else SimpleGridworld
    return constructor(num_conventions=EXPERIMENTS[experiment]["num_conventions"], **kwargs)
