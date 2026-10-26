"""Explicit agent: update a belief, then use it to choose an action.

The learner sees only its five observation values and its own last belief.
The real convention ID is NOT an input. It is used later to train the belief
network with cross-entropy loss.
"""

import torch
from torch import nn
from torch.distributions import Categorical


class ExplicitBeliefActor(nn.Module):
    def __init__(self, obs_dim=5, action_dim=5, num_conventions=3, hidden_dim=64):
        super().__init__()
        if num_conventions not in (3, 6):
            raise ValueError("num_conventions must be 3 or 6")
        self.obs_dim = obs_dim
        self.num_conventions = num_conventions
        self.action_dim = action_dim

        # MLP 1: combine the current observation and previous belief.
        self.belief_net = nn.Sequential(
            nn.Linear(obs_dim + num_conventions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_conventions),
        )

        # MLP 2: choose an action using the observation and new belief.
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim + num_conventions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
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

    def forward(self, observations, belief=None):
        """Process one step or a whole sequence in time order.

        Input: [batch, 5] or [batch, time, 5].
        Output: action logits and belief logits for EVERY step,
                and the final belief to carry into the next step.
        """
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        if observations.ndim != 3 or observations.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected [batch, time, {self.obs_dim}] observations")

        observations = observations.float()
        batch_size = observations.shape[0]
        if belief is None:
            belief = self.initial_belief(batch_size, observations.device)
        if belief.shape != (batch_size, self.num_conventions):
            raise ValueError(f"Expected belief shape ({batch_size}, {self.num_conventions})")

        action_logits = []
        belief_logits = []
        for t in range(observations.shape[1]):
            obs_t = observations[:, t, :]
            logits_b = self.belief_net(torch.cat((obs_t, belief), dim=-1))
            belief = torch.softmax(logits_b, dim=-1)
            logits_a = self.actor_net(torch.cat((obs_t, belief), dim=-1))
            action_logits.append(logits_a)
            belief_logits.append(logits_b)

        return (
            torch.stack(action_logits, dim=1),
            torch.stack(belief_logits, dim=1),
            belief,
        )

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
