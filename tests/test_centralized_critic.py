"""Checks for the critic shared by the implicit and explicit agents."""

import numpy as np
import pytest
import torch
from agents.centralized_critic import CentralizedCritic, build_critic_state
from agents.implicit_gru_agent import CentralizedCritic as ImplicitCritic


def fake_observation(convention):
    return {
        "learner": np.array([0, 4, 3, 1, 5]),
        "partner": np.array([3, 1, 0, 4, 1, 3]),
        "convention": np.array([convention]),
    }


@pytest.mark.parametrize("num_conventions,convention", [(3, 2), (6, 5)])
def test_state_and_value_shape(num_conventions, convention):
    state = build_critic_state(fake_observation(convention), 0,
                               num_conventions=num_conventions)
    assert state.shape == (12 + num_conventions,)
    assert state.dtype == np.float32
    assert np.isclose(state[11:11 + num_conventions].sum(), 1)
    assert state[11 + convention] == 1
    assert state[-1] == 0
    critic = CentralizedCritic(num_conventions=num_conventions)
    values = critic(torch.tensor(np.stack([state, state])))
    assert values.shape == (2,)
    values.mean().backward()
    assert critic.value_head[0].weight.grad is not None


def test_time_at_last_step():
    state = build_critic_state(fake_observation(1), 12, num_conventions=3)
    assert state[-1] == 1.0


def test_convention_is_not_in_local_observation():
    state = build_critic_state(fake_observation(2), 3, num_conventions=3)
    assert state[:5].shape == (5,)
    assert state.shape == (15,)


def test_old_import_still_works():
    assert ImplicitCritic is CentralizedCritic


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        build_critic_state(fake_observation(5), 1, num_conventions=3)
    with pytest.raises(ValueError):
        CentralizedCritic(global_state_dim=15, num_conventions=6)
