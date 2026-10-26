"""Shared value network and training-state builder for both MAPPO agents.

Only the critic uses this state. Actors must receive observation['learner'] only.
"""

import numpy as np
import torch
from torch import nn


def build_critic_state(observation, episode_step, max_steps=12, num_conventions=None):
    """Make one critic input: learner(5), partner(6), convention(N), time(1).

    The true convention is training-only information. Never give this vector
    to the GRU actor or the explicit belief network.
    """
    learner = np.asarray(observation["learner"], dtype=np.float32)
    partner = np.asarray(observation["partner"], dtype=np.float32)
    convention = int(np.asarray(observation["convention"]).item())
    if num_conventions is None:
        raise ValueError("Please set num_conventions to 3 or 6")
    if num_conventions not in (3, 6):
        raise ValueError("num_conventions must be 3 or 6")
    if learner.shape != (5,) or partner.shape != (6,):
        raise ValueError("Expected learner shape (5,) and partner shape (6,)")
    if not 0 <= convention < num_conventions:
        raise ValueError("Convention ID is outside this experiment")
    if max_steps <= 0 or not 0 <= episode_step <= max_steps:
        raise ValueError("Invalid episode_step or max_steps")

    convention_onehot = np.zeros(num_conventions, dtype=np.float32)
    convention_onehot[convention] = 1.0
    time = np.array([episode_step / max_steps], dtype=np.float32)
    return np.concatenate((learner, partner, convention_onehot, time))


class CentralizedCritic(nn.Module):
    """Predict the learner's future reward using training-only information."""

    def __init__(self, global_state_dim=None, hidden_dim=64, num_conventions=3):
        super().__init__()
        if num_conventions not in (3, 6):
            raise ValueError("num_conventions must be 3 or 6")
        expected_dim = 12 + num_conventions
        if global_state_dim is None:
            global_state_dim = expected_dim
        if global_state_dim != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} critic inputs for {num_conventions} conventions"
            )
        self.global_state_dim = global_state_dim
        self.value_head = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state):
        if global_state.shape[-1] != self.global_state_dim:
            raise ValueError(f"Expected critic input size {self.global_state_dim}")
        return self.value_head(global_state.float()).squeeze(-1)
