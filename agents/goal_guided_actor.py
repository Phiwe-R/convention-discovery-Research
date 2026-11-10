""" Agent 2 script: Goal-guided extension of the joint-belief MAPPO actor (joint_belief_agent.py)

Goal and convention beliefs are inferred from learner observation histories.
At action selection, the 8-way goal belief is converted into scores for movement
that shortens wraparound-grid distance to likely goals. The original neural actor
learns residual action preferences and convention-dependent adjustments.

Only the known PUBLIC candidate goal coordinates appear here. True goal/convention
IDs are never read in forward(), get_action() or evaluate_actions().
"""
import torch
from torch import nn
from .joint_belief_agent import JointBeliefActor
from environments.goal_configurations import LEARNER_GOALS


class GoalGuidedJointActor(JointBeliefActor):
    def __init__(self, *args, guide_strength=3.0, collision_penalty=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        if guide_strength < 0 or collision_penalty < 0:
            raise ValueError('Guide strength and collision penalty must be nonnegative')
        self.guide_strength = float(guide_strength)
        self.collision_penalty = float(collision_penalty)
        # Buffers are part of the checkpoint but are not trainable parameters.
        self.register_buffer('candidate_goals', torch.tensor(LEARNER_GOALS, dtype=torch.float32))
        self.register_buffer('move_delta', torch.tensor([[-1.,0.],[1.,0.],[0.,-1.],[0.,1.],[0.,0.]]))

    def guided_scores(self, observations, goal_beliefs):
        """[B,T,5] score from expected progress to candidate learner goals.

        Score(a)=E_{g~p(g|history)}[d(position,g)-d(next(position,a),g)]
                 - collision_penalty*1[next(position,a)==partner_position].
        d is Manhattan distance with the environment's 5x5 toroidal wrap.
        """
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        pos = observations[..., :2].float()
        partner_pos = observations[..., 2:4].float()
        next_positions = (pos.unsqueeze(-2) + self.move_delta) % 5  # B,T,5,2
        goals = self.candidate_goals
        before = (pos.unsqueeze(-2) - goals).abs()  # B,T,8,2
        before = torch.minimum(before, 5-before).sum(dim=-1)  # B,T,8
        after = (next_positions.unsqueeze(-2) - goals).abs()  # B,T,5,8,2
        after = torch.minimum(after, 5-after).sum(dim=-1)  # B,T,5,8
        progress = (before.unsqueeze(-2) - after) * goal_beliefs.detach().unsqueeze(-2)
        score = progress.sum(dim=-1)
        # Avoid a currently occupied partner cell when another option is open.
        # The partner may move; this is a conservative, not perfect, collision rule.
        occupied = (next_positions == partner_pos.unsqueeze(-2)).all(dim=-1)
        # Staying is not intrinsically unsafe: the partner may approach or leave.
        occupied = occupied & (self.move_delta.abs().sum(-1) > 0)
        return score - self.collision_penalty * occupied.to(score.dtype)

    def forward(self, observations, memory=None, belief_mode='real'):
        neural_logits, conv_logits, goal_logits, memory = super().forward(
            observations, memory, belief_mode=belief_mode
        )
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        goal_probs = goal_logits.softmax(-1).detach()
        if belief_mode in ('uniform', 'uniform_goal'):
            goal_probs = torch.full_like(goal_probs, 1.0 / self.num_goals)
        scores = self.guided_scores(observations, goal_probs)
        action_logits = neural_logits + self.guide_strength * scores
        return action_logits, conv_logits, goal_logits, memory

    def actor_parameters(self):
        return super().actor_parameters()
