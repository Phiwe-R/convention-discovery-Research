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
    action_logits, belief_logits, final_memory = agent(torch.randn(2, 4, 5))
    assert action_logits.shape == (2, 4, 5)
    assert belief_logits.shape == (2, 4, n)
    final_belief, hidden, action_hidden = final_memory
    assert action_hidden.shape == (1, 2, 64)
    assert hidden.shape == (1, 2, 64)
    assert final_belief.shape == (2, n)
    assert torch.allclose(final_belief.sum(-1), torch.ones(2), atol=1e-6)


def test_action_and_ppo_shapes():
    agent = ExplicitBeliefActor()
    actions, logp, entropy, belief = agent.get_action(torch.randn(2, 5))
    assert actions.shape == logp.shape == entropy.shape == (2,)
    assert belief[0].shape == (2, 3)
    assert belief[1].shape == (1, 2, 64)
    logp_seq, entropy_seq, belief_logits, final = agent.evaluate_actions(
        torch.randn(2, 4, 5), torch.randint(0, 5, (2, 4))
    )
    assert logp_seq.shape == entropy_seq.shape == (2, 4)
    assert belief_logits.shape == (2, 4, 3)
    assert final[0].shape == (2, 3)


def test_previous_hidden_state_changes_new_prediction():
    torch.manual_seed(5)
    agent = ExplicitBeliefActor()
    obs = torch.randn(1, 5)
    p1 = agent.initial_memory(1)
    p2 = (p1[0], torch.ones_like(p1[1]), p1[2])
    _, logits1, _ = agent(obs, p1)
    _, logits2, _ = agent(obs, p2)
    assert not torch.allclose(logits1, logits2)


def test_cross_entropy_updates_belief_network():
    agent = ExplicitBeliefActor(num_conventions=6)
    _, belief_logits, _ = agent(torch.randn(2, 4, 5))
    true_ids = torch.tensor([2, 5]).unsqueeze(1).expand(2, 4)
    loss = F.cross_entropy(belief_logits.reshape(-1, 6), true_ids.reshape(-1))
    loss.backward()
    assert agent.belief_net.gru.weight_ih_l0.grad is not None
    assert agent.belief_net.gru.weight_ih_l0.grad.abs().sum() > 0


def test_ppo_gradient_does_not_reach_belief_network():
    agent = ExplicitBeliefActor()
    logp, _, _, _ = agent.evaluate_actions(
        torch.randn(2, 4, 5), torch.randint(0, 5, (2, 4))
    )
    (-logp.mean()).backward()
    assert agent.actor_net.gru.weight_ih_l0.grad.abs().sum() > 0
    assert all(p.grad is None for p in agent.belief_net.parameters())


def test_belief_carries_over_and_resets():
    agent = ExplicitBeliefActor()
    a, b = torch.randn(1, 5), torch.randn(1, 5)
    with torch.no_grad():
        _, _, after_a = agent(a)
        _, carry_logits, _ = agent(b, after_a)
        _, reset_logits, _ = agent(b, agent.initial_memory(1))
        _, _, repeat_a = agent(a, agent.initial_memory(1))
    assert not torch.allclose(carry_logits, reset_logits)
    assert all(torch.allclose(x, y) for x, y in zip(after_a, repeat_a))


def test_reject_wrong_dimensions():
    agent = ExplicitBeliefActor()
    with pytest.raises(ValueError):
        agent(torch.randn(2, 4))
    with pytest.raises(ValueError):
        agent(torch.randn(2, 5), torch.randn(2, 6))


def test_action_memory_affects_policy_without_changing_belief():
    torch.manual_seed(17)
    agent = ExplicitBeliefActor()
    memory = agent.initial_memory(2)
    other_memory = (memory[0], memory[1], torch.ones_like(memory[2]))
    obs = torch.randn(2, 5)
    action1, identity1, _ = agent(obs, memory)
    action2, identity2, _ = agent(obs, other_memory)
    assert not torch.allclose(action1, action2)
    assert torch.equal(identity1, identity2)


def test_two_grus_have_disjoint_parameters():
    agent = ExplicitBeliefActor()
    belief_ids = {id(p) for p in agent.belief_net.parameters()}
    action_ids = {id(p) for p in agent.actor_net.parameters()}
    assert not belief_ids & action_ids
