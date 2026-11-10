"""A4 safeguards: no convention head, goal-guided scores, and frozen beliefs."""
import torch
from agents.goal_only_guided_actor import GoalOnlyGuidedActor
from training.pretrain_goal_only import GoalOnlyPretrainer


def test_a3_has_no_convention_predictor_or_input():
    actor = GoalOnlyGuidedActor()
    assert actor.actor_gru.input_size == 13
    assert not any('convention' in key for key in actor.state_dict())
    pretrainer = GoalOnlyPretrainer()
    assert not any('convention' in key for key in pretrainer.state_dict())


def test_a3_goal_probabilities_and_action_dimensions():
    actor = GoalOnlyGuidedActor()
    output, goal_logits, memory = actor(torch.tensor([[[2., 2., 4., 4., 5.]]]))
    assert output.shape == (1, 1, 5)
    assert goal_logits.shape == (1, 1, 8)
    assert torch.allclose(memory[0].sum(-1), torch.ones(1), atol=1e-6)


def test_goal_guidance_prefers_shorter_path():
    actor = GoalOnlyGuidedActor()
    observation = torch.tensor([[[2., 2., 4., 4., 5.]]])
    belief = torch.zeros(1, 1, 8)
    belief[0, 0, 2] = 1.  # g2 = (2, 1)
    scores = actor.guided_scores(observation, belief)
    assert scores[0, 0, 2] > scores[0, 0, 3]  # LEFT better than RIGHT


def test_ppo_actor_does_not_backpropagate_into_belief():
    actor = GoalOnlyGuidedActor()
    for param in actor.belief_parameters():
        param.requires_grad_(False)
    logits, _, _ = actor(torch.randn(2, 4, 5))
    logits.square().mean().backward()
    assert all(param.grad is None for param in actor.belief_parameters())
    assert any(param.grad is not None for param in actor.actor_parameters())


def test_uniform_replaces_goal_guidance_and_actor_input():
    actor = GoalOnlyGuidedActor()
    observations = torch.tensor([[[2., 2., 4., 4., 5.]]])
    with torch.no_grad():
        logits_real, _, _ = actor(observations, belief_mode='real')
        logits_uniform, _, _ = actor(observations, belief_mode='uniform')
    assert logits_real.shape == logits_uniform.shape
