"""Tests for the pure implicit GRU and the configurable centralized critic."""
import pytest
import torch

from agents.implicit_gru_agent import GRUActor, CentralizedCritic


def test_actor_sequence_and_actions():
    actor = GRUActor()
    x = torch.randn(2, 4, 5)
    logits, memory = actor(x)
    assert logits.shape == (2, 4, 5)
    assert memory.shape == (1, 2, 64)
    action, logp, entropy, next_memory = actor.get_action(x)
    assert action.shape == logp.shape == entropy.shape == (2,)
    assert next_memory.shape == (1, 2, 64)
    assert torch.all((action >= 0) & (action < 5))


def test_actor_has_no_identification_head():
    actor = GRUActor()
    assert not hasattr(actor, 'identification_head')
    assert not hasattr(actor, 'belief_head')
    assert actor.gru.input_size == 5


def test_recurrent_memory_changes_with_history_and_resets():
    torch.manual_seed(8)
    actor = GRUActor()
    a = torch.randn(1, 5)
    b = torch.randn(1, 5)
    with torch.no_grad():
        initial = actor.initial_hidden(1)
        _, after_a = actor(a, initial)
        logits_carried, _ = actor(b, after_a)
        logits_reset, _ = actor(b, actor.initial_hidden(1))
        _, fresh = actor(a, actor.initial_hidden(1))
    assert not torch.allclose(logits_carried, logits_reset)
    assert torch.allclose(after_a, fresh)


def test_actor_sequence_gradients():
    actor = GRUActor()
    logits, _ = actor(torch.randn(2, 4, 5))
    logits.square().mean().backward()
    assert actor.gru.weight_ih_l0.grad is not None
    assert actor.policy_head[0].weight.grad is not None


def test_evaluate_actions_matches_shapes_and_backprop():
    actor = GRUActor()
    x = torch.randn(2, 4, 5)
    a = torch.randint(0, 5, (2, 4))
    logp, entropy, memory = actor.evaluate_actions(x, a)
    assert logp.shape == entropy.shape == (2, 4)
    assert memory.shape == (1, 2, 64)
    (-logp.mean() - .01 * entropy.mean()).backward()
    assert actor.gru.weight_ih_l0.grad is not None


@pytest.mark.parametrize('count,dim', [(3, 15), (6, 18)])
def test_critic_for_each_experiment(count, dim):
    critic = CentralizedCritic(num_conventions=count)
    assert critic.global_state_dim == dim
    values = critic(torch.randn(2, dim))
    assert values.shape == (2,)
    values.mean().backward()
    assert critic.value_head[0].weight.grad is not None


def test_critic_rejects_inconsistent_dimension():
    with pytest.raises(ValueError):
        CentralizedCritic(global_state_dim=15, num_conventions=6)
