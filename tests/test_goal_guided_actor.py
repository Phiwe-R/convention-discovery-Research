"""Tests for goal-guided action selection and gradient isolation."""
import torch
from agents.goal_guided_actor import GoalGuidedJointActor
from training.train_goal_guided import oracle_move_loss


def test_guidance_favors_shortest_route():
    actor = GoalGuidedJointActor(num_conventions=6)
    observations = torch.tensor([[[1., 0., 4., 4., 5.]]])
    beliefs = torch.nn.functional.one_hot(torch.tensor([[0]]), num_classes=8).float()
    scores = actor.guided_scores(observations, beliefs)[0,0]
    assert scores.argmax().item() == 0  # UP toward goal 0=(0,0)
    assert scores[0] > scores[1]
    assert scores[0] > scores[4]


def test_toroidal_shortest_path():
    actor = GoalGuidedJointActor(num_conventions=6)
    observations = torch.tensor([[[4., 0., 2., 2., 5.]]])
    beliefs = torch.nn.functional.one_hot(torch.tensor([[0]]), num_classes=8).float()
    scores = actor.guided_scores(observations, beliefs)[0,0]
    assert scores.argmax().item() == 1  # DOWN wraps row 4 to row 0


def test_uniform_belief_ablation_affects_guidance():
    actor = GoalGuidedJointActor(num_conventions=6, guide_strength=3.0)
    obs = torch.tensor([[[1.,0.,4.,4.,5.]]])
    goal_probs = torch.nn.functional.one_hot(torch.tensor([[0]]), num_classes=8).float()
    real_scores = actor.guided_scores(obs, goal_probs)
    uniform = torch.full_like(goal_probs, 1.0/8)
    uniform_scores = actor.guided_scores(obs, uniform)
    assert not torch.allclose(real_scores, uniform_scores)


def test_imitation_loss_decreases_when_correct_action_strengthened():
    actor = GoalGuidedJointActor(num_conventions=6)
    obs = torch.tensor([[[1.,0.,4.,4.,5.]]])
    baseline = torch.zeros(1,1,5)
    better = baseline.clone(); better[0,0,0] += 4
    assert oracle_move_loss(actor, obs, better, 0) < oracle_move_loss(actor, obs, baseline, 0)


def test_ppo_does_not_update_belief_network():
    torch.manual_seed(3)
    actor = GoalGuidedJointActor(num_conventions=6)
    obs = torch.tensor([[[1.,0.,4.,4.,5.], [0.,0.,4.,4.,0.]]])
    logits, _, _, _ = actor(obs)
    loss = -torch.log_softmax(logits,dim=-1)[0,0,0]
    loss.backward()
    assert actor.action_head.weight.grad is not None
    assert all(p.grad is None for p in actor.belief_parameters())


def test_goal_probabilities_are_normalized():
    actor=GoalGuidedJointActor(num_conventions=6)
    logits, cl, gl, memory=actor(torch.zeros(2,4,5))
    assert logits.shape == (2,4,5)
    assert cl.shape == (2,4,6)
    assert gl.shape == (2,4,8)
    assert torch.allclose(gl.softmax(-1).sum(-1),torch.ones(2,4))
