"""Agent 4 : Goal belief Only 

Actor inputs: learner-local observation (5), predicted goal belief (8).
No convention head, no convention logits, no true goal at action selection.
The convention is still hidden in the environment, and is visible only to the
existing centralized training critic.
"""
import torch
from torch import nn
from torch.distributions import Categorical
from environments.goal_configurations import LEARNER_GOALS


class GoalOnlyGuidedActor(nn.Module):
    def __init__(self, obs_dim=5, action_dim=5, num_goals=8, hidden_dim=64,
                 guide_strength=5.0, collision_penalty=1.5):
        super().__init__()
        if num_goals != len(LEARNER_GOALS) or action_dim != 5 or obs_dim != 5:
            raise ValueError('A3 requires 5 observation inputs, 8 candidate goals and 5 actions')
        self.obs_dim = obs_dim
        self.num_goals = num_goals
        self.hidden_dim = hidden_dim
        self.guide_strength = float(guide_strength)
        self.collision_penalty = float(collision_penalty)
        if self.guide_strength < 0 or self.collision_penalty < 0:
            raise ValueError('Guide and collision penalty must be nonnegative')

        self.belief_gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.goal_head = nn.Linear(hidden_dim, num_goals)
        self.actor_gru = nn.GRU(obs_dim + num_goals, hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer('candidate_goals', torch.tensor(LEARNER_GOALS, dtype=torch.float32))
        self.register_buffer('move_delta', torch.tensor([[-1., 0.], [1., 0.], [0., -1.], [0., 1.], [0., 0.]]))

    def belief_parameters(self):
        return list(self.belief_gru.parameters()) + list(self.goal_head.parameters())

    def actor_parameters(self):
        return list(self.actor_gru.parameters()) + list(self.action_head.parameters())

    def initial_memory(self, batch_size, device=None):
        ref = next(self.parameters())
        device = device if device is not None else ref.device
        belief = torch.full((batch_size, self.num_goals), 1 / self.num_goals, device=device, dtype=ref.dtype)
        h_b = torch.zeros(1, batch_size, self.hidden_dim, device=device, dtype=ref.dtype)
        return (belief, h_b, h_b.clone())

    def guided_scores(self, observations, goal_beliefs):
        """Expected reduction in toroidal Manhattan distance under P(goal|history)."""
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        pos = observations[..., :2].float()
        partner = observations[..., 2:4].float()
        nxt = (pos.unsqueeze(-2) + self.move_delta) % 5
        before_delta = (pos.unsqueeze(-2) - self.candidate_goals).abs()
        before = torch.minimum(before_delta, 5 - before_delta).sum(-1)
        after_delta = (nxt.unsqueeze(-2) - self.candidate_goals).abs()
        after = torch.minimum(after_delta, 5 - after_delta).sum(-1)
        score = ((before.unsqueeze(-2) - after) * goal_beliefs.detach().unsqueeze(-2)).sum(-1)
        occupied = (nxt == partner.unsqueeze(-2)).all(-1)
        occupied = occupied & (self.move_delta.abs().sum(-1) > 0)
        return score - self.collision_penalty * occupied.to(score.dtype)

    def forward(self, observations, memory=None, belief_mode='real'):
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        if observations.ndim != 3 or observations.shape[-1] != self.obs_dim:
            raise ValueError(f'Expected [batch, time, {self.obs_dim}] observations')
        if belief_mode not in ('real', 'uniform', 'uniform_goal'):
            raise ValueError('A3 accepts real, uniform, uniform_goal (no convention belief exists)')
        observations = observations.float()
        if memory is None:
            memory = self.initial_memory(observations.shape[0], observations.device)
        _, h_b, h_a = memory
        features, h_b = self.belief_gru(observations, h_b)
        goal_logits = self.goal_head(features)
        goal_probs = goal_logits.softmax(-1)
        actor_goal = goal_probs.detach() if belief_mode == 'real' else torch.full_like(goal_probs, 1. / self.num_goals)
        actor_input = torch.cat((observations, actor_goal), dim=-1)
        actor_features, h_a = self.actor_gru(actor_input, h_a)
        neural_logits = self.action_head(actor_features)
        scores = self.guided_scores(observations, actor_goal)
        action_logits = neural_logits + self.guide_strength * scores
        return action_logits, goal_logits, (goal_probs[:, -1], h_b, h_a)

    def get_action(self, observation, memory=None, deterministic=False, belief_mode='real'):
        logits, goal_logits, memory = self.forward(observation, memory, belief_mode)
        distribution = Categorical(logits=logits[:, -1])
        action = logits[:, -1].argmax(-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), memory

    def evaluate_actions(self, observations, actions, belief_mode='real'):
        logits, goal_logits, _ = self.forward(observations, belief_mode=belief_mode)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), goal_logits
