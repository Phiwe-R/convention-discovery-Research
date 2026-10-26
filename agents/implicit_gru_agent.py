"""Pure implicit recurrent actor and centralized critic for MAPPO.

The actor NEVER receives the convention ID, goals, or global state.
No convention classifier or cross-entropy loss is part of this architecture.
"""

import torch
from torch import nn
from torch.distributions import Categorical

# Use the same critic for the implicit and explicit agents.
from agents.centralized_critic import CentralizedCritic


class GRUActor(nn.Module):
    """Recurrent learner policy using only the 5-element local observation.

    forward accepts [batch, obs_dim] for one step or
    [batch, sequence_length, obs_dim] for recurrent PPO training.
    Returns action logits for EVERY timestep, and the final hidden state.
    """

    def __init__(self, obs_dim=5, action_dim=5, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def initial_hidden(self, batch_size, device=None):
        """Use at episode reset; clear only the finished environments in a batch."""
        param = next(self.parameters())
        return torch.zeros(1, batch_size, self.hidden_dim,
                           dtype=param.dtype, device=device or param.device)

    def forward(self, observations, hidden=None):
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        if observations.ndim != 3 or observations.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected [batch, time, {self.obs_dim}] or [batch, {self.obs_dim}], "
                f"got {tuple(observations.shape)}"
            )
        output, next_hidden = self.gru(observations.float(), hidden)
        return self.policy_head(output), next_hidden

    def get_action(self, observations, hidden=None, deterministic=False):
        """Sample/select last-timestep action; return PPO log-prob and entropy."""
        logits, next_hidden = self.forward(observations, hidden)
        distribution = Categorical(logits=logits[:, -1, :])
        action = (logits[:, -1, :].argmax(dim=-1)
                  if deterministic else distribution.sample())
        return (action, distribution.log_prob(action),
                distribution.entropy(), next_hidden)

    def evaluate_actions(self, observations, actions, hidden=None):
        """Evaluate an action at each timestep of a PPO training sequence.

        Returns log_probs, entropy [batch, time], and final hidden state.
        Call on sequences with properly reset/detached initial memory.
        """
        logits, next_hidden = self.forward(observations, hidden)
        distribution = Categorical(logits=logits)
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        return (distribution.log_prob(actions.long()),
                distribution.entropy(), next_hidden)


