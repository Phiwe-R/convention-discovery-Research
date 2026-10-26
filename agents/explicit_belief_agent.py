"""Explicit agent: update a belief, then use it to choose an action.

The belief GRU sees only the five observation values and its recurrent memory.
The real convention ID is NOT an input. It is used later to train the belief
network with cross-entropy loss.
"""

import torch
from torch import nn
from torch.distributions import Categorical


class RecurrentBeliefNetwork(nn.Module):
    """Identify the convention from a sequence; trained only by identification CE."""

    def __init__(self, obs_dim, hidden_dim, num_conventions):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.identity_head = nn.Linear(hidden_dim, num_conventions)

    def forward(self, observations, hidden):
        features, hidden = self.gru(observations, hidden)
        return self.identity_head(features), hidden


class RecurrentActionNetwork(nn.Module):
    """Choose actions using observations, detached beliefs, and private memory."""

    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, inputs, hidden):
        features, hidden = self.gru(inputs, hidden)
        return self.action_head(features), hidden


class ExplicitBeliefActor(nn.Module):
    def __init__(self, obs_dim=5, action_dim=5, num_conventions=3, hidden_dim=64):
        super().__init__()
        if num_conventions not in (3, 6):
            raise ValueError("num_conventions must be 3 or 6")
        self.obs_dim = obs_dim
        self.num_conventions = num_conventions
        self.action_dim = action_dim

        # Standalone identifier: observation history -> convention probabilities.
        self.hidden_dim = hidden_dim
        self.belief_net = RecurrentBeliefNetwork(obs_dim, hidden_dim, num_conventions)

        # Independent recurrent policy; no hidden state is shared with the identifier.
        self.actor_net = RecurrentActionNetwork(
            obs_dim + num_conventions, hidden_dim, action_dim
        )

    def initial_belief(self, batch_size, device=None):
        """Start each episode with equal probability for each convention."""
        param = next(self.parameters())
        return torch.full(
            (batch_size, self.num_conventions),
            1.0 / self.num_conventions,
            dtype=param.dtype,
            device=device if device is not None else param.device,
        )

    def initial_memory(self, batch_size, device=None):
        """Reset the uniform belief and both independent GRU states."""
        belief = self.initial_belief(batch_size, device)
        hidden = belief.new_zeros(1, batch_size, self.hidden_dim)
        return belief, hidden, torch.zeros_like(hidden)

    def forward(self, observations, belief=None):
        """Return action logits, identity logits, and (belief, belief GRU state, action GRU state).

        `belief` is the complete memory tuple from the preceding call.
        Pass None at an episode boundary to reset it. The actor consumes
        detached identity probabilities, never the private GRU state.
        """
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        if observations.ndim != 3 or observations.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected [batch, time, {self.obs_dim}] observations")
        observations = observations.float()
        batch_size = observations.shape[0]
        if belief is None:
            belief = self.initial_memory(batch_size, observations.device)
        if not isinstance(belief, tuple) or len(belief) != 3:
            raise ValueError("Expected (belief, belief GRU state, action GRU state) from initial_memory or forward")
        previous_belief, hidden, action_hidden = belief
        if previous_belief.shape != (batch_size, self.num_conventions):
            raise ValueError("Invalid belief shape")
        if hidden.shape != (1, batch_size, self.hidden_dim):
            raise ValueError("Invalid belief GRU state shape")

        if action_hidden.shape != (1, batch_size, self.hidden_dim):
            raise ValueError("Invalid action GRU state shape")

        belief_logits, hidden = self.belief_net(observations, hidden)
        probabilities = torch.softmax(belief_logits, dim=-1)
        action_logits, action_hidden = self.actor_net(
            torch.cat((observations, probabilities.detach()), dim=-1), action_hidden
        )
        return action_logits, belief_logits, (probabilities[:, -1, :], hidden, action_hidden)

    def get_action(self, observation, belief=None, deterministic=False):
        """Take an action and return the updated belief for the next step."""
        action_logits, belief_logits, next_belief = self.forward(observation, belief)
        distribution = Categorical(logits=action_logits[:, -1, :])
        action = (action_logits[:, -1, :].argmax(dim=-1)
                  if deterministic else distribution.sample())
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            next_belief,
        )

    def evaluate_actions(self, observations, actions, belief=None):
        """Compute PPO values and convention logits for training sequences."""
        action_logits, belief_logits, next_belief = self.forward(observations, belief)
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        distribution = Categorical(logits=action_logits)
        return (
            distribution.log_prob(actions.long()),
            distribution.entropy(),
            belief_logits,
            next_belief,
        )
