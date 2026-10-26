"""Check how the explicit belief network learns during MAPPO training.

Run from your project root:
    python -m pytest tests/test_belief_training_diagnostics.py -v

These tests do not change your agent or train a full model.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.explicit_belief_agent import ExplicitBeliefActor
from training.train_mappo import create_models, collect_episode, update_models
from tests.test_train_mappo import SmallEnv


def make_actor(num_conventions=3):
    torch.manual_seed(123)
    return ExplicitBeliefActor(obs_dim=5, action_dim=5,
                               num_conventions=num_conventions, hidden_dim=64)


def has_nonzero_gradient(parameters):
    return any(p.grad is not None and p.grad.abs().sum().item() > 1e-10
               for p in parameters)


def test_ce_gradients_reach_belief_mlp():
    """CE must update the belief network, not the action network."""
    actor = make_actor()
    observations = torch.randn(2, 5, 5)
    _, belief_logits, _ = actor(observations)
    true_conventions = torch.tensor([1, 2]).repeat_interleave(5)
    ce_loss = F.cross_entropy(belief_logits.reshape(-1, 3), true_conventions)
    actor.zero_grad()
    ce_loss.backward()
    assert has_nonzero_gradient(actor.belief_net.parameters())
    assert not has_nonzero_gradient(actor.actor_net.parameters())


def test_actor_loss_does_not_reach_belief_network():
    """PPO trains the actor without changing its standalone identifier."""
    actor = make_actor()
    observations = torch.randn(2, 4, 5)
    action_logits, _, _ = actor(observations)
    actions = torch.tensor([[0, 2, 4, 1], [3, 1, 0, 2]])
    policy_loss = -Categorical(logits=action_logits).log_prob(actions).mean()
    actor.zero_grad()
    policy_loss.backward()
    assert has_nonzero_gradient(actor.actor_net.parameters())
    assert not has_nonzero_gradient(actor.belief_net.parameters())


def test_full_sequence_matches_step_by_step():
    """Beliefs must be carried from one observation to the next."""
    actor = make_actor(6)
    observations = torch.randn(2, 7, 5)
    with torch.no_grad():
        full_actions, full_belief_logits, full_belief = actor(observations)
        belief = actor.initial_memory(2)
        individual_actions, individual_beliefs = [], []
        for t in range(7):
            action_logits, belief_logits, belief = actor(observations[:, t, :], belief)
            individual_actions.append(action_logits)
            individual_beliefs.append(belief_logits)
    assert torch.allclose(full_actions, torch.cat(individual_actions, dim=1), atol=1e-6)
    assert torch.allclose(full_belief_logits, torch.cat(individual_beliefs, dim=1), atol=1e-6)
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(full_belief, belief))


def test_initial_belief_is_uniform_and_episode_reset_is_clean():
    """A new episode must not inherit the previous episode's belief."""
    actor = make_actor(6)
    observations = torch.randn(1, 6, 5)
    with torch.no_grad():
        initial = actor.initial_belief(1)
        assert torch.allclose(initial, torch.full((1, 6), 1 / 6))
        initial_actions, initial_logits, _ = actor(observations)
        _, _, old_belief = actor(torch.randn(1, 5, 5))
        restarted_actions, restarted_logits, _ = actor(observations, actor.initial_memory(1))
        continued_actions, _, _ = actor(observations, old_belief)
    assert torch.allclose(initial_actions, restarted_actions, atol=1e-6)
    assert torch.allclose(initial_logits, restarted_logits, atol=1e-6)
    assert not torch.allclose(initial_actions, continued_actions, atol=1e-7)


def test_true_convention_is_only_a_training_label():
    """Changing a CE target cannot change the actor's forward output."""
    actor = make_actor()
    obs = torch.randn(1, 4, 5)
    with torch.no_grad():
        actions_before, beliefs_before, _ = actor(obs)
        actions_after, beliefs_after, _ = actor(obs)
    assert torch.equal(actions_before, actions_after)
    assert torch.equal(beliefs_before, beliefs_after)
    assert F.cross_entropy(beliefs_before[:, -1, :], torch.tensor([0])) != F.cross_entropy(
        beliefs_before[:, -1, :], torch.tensor([1]))


def test_training_update_changes_belief_weights():
    """The full PPO + CE update must really change belief-network weights."""
    torch.manual_seed(42)
    actor, critic = create_models('explicit', 3, torch.device('cpu'))
    episodes = [collect_episode(SmallEnv(), actor, critic, 'explicit', 3,
                                torch.device('cpu')) for _ in range(4)]
    before = [p.detach().clone() for p in actor.belief_net.parameters()]
    optimiser = torch.optim.Adam(list(actor.actor_net.parameters()) + list(critic.parameters()), lr=2.5e-4)
    belief_optimiser = torch.optim.Adam(actor.belief_net.parameters(), lr=2.5e-4)
    info = update_models(actor, critic, optimiser, episodes, 'explicit', epochs=2, ce_coef=0.1, belief_optimizer=belief_optimiser)
    assert info['belief_ce'] is not None and info['belief_ce'] > 0
    assert any(not torch.equal(old, p.detach())
               for old, p in zip(before, actor.belief_net.parameters()))
