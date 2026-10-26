
"""Small tests for recurrent PPO, without needing to run a full training job."""

import numpy as np
import pytest
import torch

from training.train_mappo import (
    collect_episode,
    create_models,
    discounted_gae,
    train,
    update_models,
)


class SmallEnv:
    """A tiny example environment with the same wrapper interface."""

    def __init__(self, terminal=False):
        self.env = self
        self.max_steps = 3
        self.terminal = terminal
        self.index = 0

    def observation(self):
        return {
            "learner": np.array([0, 0, 1, 1, 5], dtype=np.int32),
            "partner": np.array([1, 1, 0, 0, 2, 2], dtype=np.int32),
            "convention": np.array([1], dtype=np.int32),
        }

    def reset(self, seed=None):
        self.index = 0
        return self.observation(), {}

    def step(self, action):
        self.index += 1
        done = self.terminal and self.index == 2
        return (
            self.observation(),
            -0.01,
            done,
            self.index == 3 and not done,
            {},
        )

    def close(self):
        pass


def test_gae_bootstrap_on_truncation():
    rewards = torch.tensor([0.0, 0.0])
    values = torch.zeros(2)

    _, terminated_return = discounted_gae(
        rewards, values, torch.tensor(2.0), True,
        gamma=1.0, gae_lambda=1.0,
    )

    _, truncated_return = discounted_gae(
        rewards, values, torch.tensor(2.0), False,
        gamma=1.0, gae_lambda=1.0,
    )

    assert torch.allclose(terminated_return, torch.zeros(2))
    assert torch.allclose(truncated_return, torch.full((2,), 2.0))


@pytest.mark.parametrize("agent_name", ["implicit", "explicit"])
def test_episodes_and_ppo_update(agent_name):
    torch.manual_seed(4)
    actor, critic = create_models(agent_name, 3, torch.device("cpu"))

    episodes = [
        collect_episode(
            SmallEnv(terminal=bool(i % 2)),
            actor,
            critic,
            agent_name,
            3,
            torch.device("cpu"),
        )
        for i in range(4)
    ]

    assert {ep["length"] for ep in episodes} == {2, 3}
    assert any(ep["truncated"] for ep in episodes)

    old = next(actor.parameters()).detach().clone()

    optimizer = torch.optim.Adam(
        list(actor.actor_net.parameters() if agent_name == "explicit" else actor.parameters()) + list(critic.parameters()),
        lr=0.01,
    )

    belief_optimizer = (torch.optim.Adam(actor.belief_net.parameters(), lr=0.01)
                        if agent_name == "explicit" else None)
    result = update_models(
        actor, critic, optimizer, episodes, agent_name, epochs=2,
        belief_optimizer=belief_optimizer
    )

    assert torch.isfinite(torch.tensor(result["policy_loss"]))
    assert not torch.allclose(old, next(actor.parameters()).detach())
    assert (result["belief_ce"] is None) == (agent_name == "implicit")


def test_short_training_checkpoint(tmp_path):
    path = train(
        timesteps=8,
        episodes_per_update=2,
        update_epochs=1,
        output_dir=str(tmp_path),
        env_factory=lambda _: SmallEnv(),
    )

    saved = torch.load(path, weights_only=True)

    assert saved["agent"] == "implicit"
    assert saved["timesteps"] >= 8


def test_explicit_training_checkpoint(tmp_path):
    path = train(
        agent_name="explicit",
        experiment="experiment2",
        timesteps=6,
        episodes_per_update=2,
        update_epochs=1,
        output_dir=str(tmp_path),
        env_factory=lambda _: SmallEnv(),
    )

    saved = torch.load(path, weights_only=True)

    assert saved["agent"] == "explicit"
    assert saved["num_conventions"] == 6
    assert saved["goal_success"] == 0.0


def test_explicit_episode_records_identification():
    actor, critic = create_models(
        "explicit", 3, torch.device("cpu")
    )

    episode = collect_episode(
        SmallEnv(),
        actor,
        critic,
        "explicit",
        3,
        torch.device("cpu"),
    )

    assert len(episode["id_correct_steps"]) == episode["length"]
    assert episode["id_correct_final"] == episode["id_correct_steps"][-1]
    assert all(value in (0, 1) for value in episode["id_correct_steps"])


def test_implicit_has_no_identification_metric():
    actor, critic = create_models(
        "implicit", 3, torch.device("cpu")
    )

    episode = collect_episode(
        SmallEnv(),
        actor,
        critic,
        "implicit",
        3,
        torch.device("cpu"),
    )

    assert episode["id_correct_final"] is None
    assert episode["id_correct_steps"] == []


def test_training_prints_recent_metrics(tmp_path, capsys):
    train(
        agent_name="explicit",
        timesteps=6,
        episodes_per_update=2,
        update_epochs=1,
        output_dir=str(tmp_path),
        env_factory=lambda _: SmallEnv(),
    )

    output = capsys.readouterr().out

    assert "Recent goal success" in output
    assert "Recent ID final" in output
    assert "Recent ID steps" in output
    assert "CE " in output
