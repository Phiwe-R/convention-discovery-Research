"""Checks for the explicit belief-updating actor."""
import pytest
import torch
import torch.nn.functional as F

from agents.explicit_belief_agent import ExplicitBeliefActor


@pytest.mark.parametrize("n", [3, 6])
def test_shapes_and_initial_belief(n):
    agent = ExplicitBeliefActor(num_conventions=n)
    initial = agent.initial_belief(2)
    assert initial.shape == (2, n)
    assert torch.allclose(initial.sum(-1), torch.ones(2))
    action_logits, belief_logits, final_belief = agent(torch.randn(2, 4, 5))
    assert action_logits.shape == (2, 4, 5)
    assert belief_logits.shape == (2, 4, n)
    assert final_belief.shape == (2, n)
    assert torch.allclose(final_belief.sum(-1), torch.ones(2), atol=1e-6)


def test_action_and_ppo_shapes():
    agent = ExplicitBeliefActor()
    actions, logp, entropy, belief = agent.get_action(torch.randn(2, 5))
    assert actions.shape == logp.shape == entropy.shape == (2,)
    assert belief.shape == (2, 3)
    logp_seq, entropy_seq, belief_logits, final = agent.evaluate_actions(
        torch.randn(2, 4, 5), torch.randint(0, 5, (2, 4))
    )
    assert logp_seq.shape == entropy_seq.shape == (2, 4)
    assert belief_logits.shape == (2, 4, 3)
    assert final.shape == (2, 3)


def test_previous_belief_changes_new_prediction():
    torch.manual_seed(5)
    agent = ExplicitBeliefActor()
    obs = torch.randn(1, 5)
    p1 = torch.tensor([[1., 0., 0.]])
    p2 = torch.tensor([[0., 1., 0.]])
    _, logits1, _ = agent(obs, p1)
    _, logits2, _ = agent(obs, p2)
    assert not torch.allclose(logits1, logits2)


def test_cross_entropy_updates_belief_network():
    agent = ExplicitBeliefActor(num_conventions=6)
    _, belief_logits, _ = agent(torch.randn(2, 4, 5))
    true_ids = torch.tensor([2, 5]).unsqueeze(1).expand(2, 4)
    loss = F.cross_entropy(belief_logits.reshape(-1, 6), true_ids.reshape(-1))
    loss.backward()
    assert agent.belief_net[0].weight.grad is not None
    assert agent.belief_net[0].weight.grad.abs().sum() > 0


def test_ppo_gradient_reaches_belief_network():
    agent = ExplicitBeliefActor()
    logp, _, _, _ = agent.evaluate_actions(
        torch.randn(2, 4, 5), torch.randint(0, 5, (2, 4))
    )
    (-logp.mean()).backward()
    assert agent.actor_net[0].weight.grad.abs().sum() > 0
    assert agent.belief_net[0].weight.grad.abs().sum() > 0


def test_belief_carries_over_and_resets():
    agent = ExplicitBeliefActor()
    a, b = torch.randn(1, 5), torch.randn(1, 5)
    with torch.no_grad():
        _, _, after_a = agent(a)
        _, carry_logits, _ = agent(b, after_a)
        _, reset_logits, _ = agent(b, agent.initial_belief(1))
        _, _, repeat_a = agent(a, agent.initial_belief(1))
    assert not torch.allclose(carry_logits, reset_logits)
    assert torch.allclose(after_a, repeat_a)


def test_reject_wrong_dimensions():
    agent = ExplicitBeliefActor()
    with pytest.raises(ValueError):
        agent(torch.randn(2, 4))
    with pytest.raises(ValueError):
        agent(torch.randn(2, 5), torch.randn(2, 6))
