"""Agent 2  : joint convention-and-goal belief agent.

Only the belief GRU sees learner observations. The true convention and hidden goal
labels are used in supervised losses, never given as actor/predictor inputs.
The policy receives predicted beliefs with PPO gradients blocked by detach().
"""

import torch
from torch import nn
from torch.distributions import Categorical


class JointBeliefActor(nn.Module):
    def __init__(self, obs_dim=5, action_dim=5, num_conventions=6,
                 num_goals=8, hidden_dim=64):
        super().__init__()
        if num_conventions not in (3, 6) or num_goals != 8:
            raise ValueError('Expected 3/6 conventions and 8 learner goals')
        self.obs_dim = obs_dim
        self.num_conventions = num_conventions
        self.num_goals = num_goals
        self.hidden_dim = hidden_dim
        self.belief_gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.convention_head = nn.Linear(hidden_dim, num_conventions)
        self.goal_head = nn.Linear(hidden_dim, num_goals)
        # Preserve the supervisor's recurrent action policy for a controlled ablation.
        self.actor_gru = nn.GRU(obs_dim + num_conventions + num_goals,
                                hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def predict_beliefs(self, observations):
        """Supervised-only forward; never runs policy network."""
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        features, _ = self.belief_gru(observations.float())
        return self.convention_head(features), self.goal_head(features)

    def initial_memory(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        ref = next(self.parameters())
        cb = torch.full((batch_size, self.num_conventions),
                        1.0 / self.num_conventions, device=device, dtype=ref.dtype)
        gb = torch.full((batch_size, self.num_goals),
                        1.0 / self.num_goals, device=device, dtype=ref.dtype)
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device, dtype=ref.dtype)
        return cb, gb, h, h.clone()

    def belief_parameters(self):
        return list(self.belief_gru.parameters()) + list(self.convention_head.parameters()) + list(self.goal_head.parameters())

    def actor_parameters(self):
        return list(self.actor_gru.parameters()) + list(self.action_head.parameters())

    def forward(self, observations, memory=None, belief_mode='real'):
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        if observations.ndim != 3 or observations.shape[-1] != self.obs_dim:
            raise ValueError(f'Expected [batch, time, {self.obs_dim}]')
        obs = observations.float()
        batch, time, _ = obs.shape
        if memory is None:
            memory = self.initial_memory(batch, obs.device)
        _, _, h_b, h_a = memory
        features, h_b = self.belief_gru(obs, h_b)
        convention_logits = self.convention_head(features)
        goal_logits = self.goal_head(features)
        cb = convention_logits.softmax(-1)
        gb = goal_logits.softmax(-1)
        if belief_mode == 'uniform':
            actor_cb = torch.full_like(cb, 1.0 / self.num_conventions)
            actor_gb = torch.full_like(gb, 1.0 / self.num_goals)
        elif belief_mode == 'uniform_convention':
            actor_cb = torch.full_like(cb, 1.0 / self.num_conventions)
            actor_gb = gb.detach()
        elif belief_mode == 'uniform_goal':
            actor_cb = cb.detach()
            actor_gb = torch.full_like(gb, 1.0 / self.num_goals)
        elif belief_mode == 'real':
            actor_cb, actor_gb = cb.detach(), gb.detach()
        else:
            raise ValueError(f'Unknown belief mode: {belief_mode}')
        actor_inputs = torch.cat((obs, actor_cb, actor_gb), dim=-1)
        act_features, h_a = self.actor_gru(actor_inputs, h_a)
        action_logits = self.action_head(act_features)
        return action_logits, convention_logits, goal_logits, (cb[:, -1], gb[:, -1], h_b, h_a)

    def get_action(self, observation, memory=None, deterministic=False, belief_mode='real'):
        action_logits, conv_logits, goal_logits, memory = self.forward(observation, memory, belief_mode)
        dist = Categorical(logits=action_logits[:, -1])
        action = action_logits[:, -1].argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), memory

    def evaluate_actions(self, observations, actions):
        action_logits, conv_logits, goal_logits, memory = self.forward(observations)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        distribution = Categorical(logits=action_logits)
        return distribution.log_prob(actions), distribution.entropy(), conv_logits, goal_logits
