"""Fast checks for evaluation; no long training run needed."""
import pytest
import torch

from evaluation.evaluate_mappo import evaluate, load_actor, save_records
from training.train_mappo import create_models


class TinyEnv:
    def __init__(self):
        self.i = 0

    def reset(self, seed=None):
        self.i = 0
        return self.observation(), {}

    def observation(self):
        import numpy as np
        return {
            "learner": np.array([0, 0, 1, 1, 5], dtype=np.int32),
            "partner": np.array([1, 1, 0, 0, 2, 2], dtype=np.int32),
            "convention": np.array([1], dtype=np.int32),
        }

    def step(self, action):
        self.i += 1
        done = self.i == 3
        return self.observation(), 1.0 if done else -0.01, done, False, {"task_success": done}

    def close(self):
        pass


@pytest.mark.parametrize("agent", ["implicit", "explicit"])
def test_evaluate(agent):
    actor, _ = create_models(agent, 3, torch.device("cpu"))
    summary, records = evaluate(actor, lambda _: TinyEnv(), "experiment1", agent, episodes=3)
    assert summary["goal_success_pct"] == 100
    assert summary["mean_length"] == 3
    assert len(records) == 3
    if agent == "explicit":
        assert 1 in summary["id_accuracy_by_step_pct"]
        uniform, _ = evaluate(actor, lambda _: TinyEnv(), "experiment1", agent, episodes=3, belief_mode="uniform")
        assert "id_accuracy_final_pct" not in uniform


def test_checkpoint_loading(tmp_path):
    actor, _ = create_models("explicit", 3, torch.device("cpu"))
    path = tmp_path / "model.pt"
    torch.save({"actor": actor.state_dict(), "experiment": "experiment1",
                "agent": "explicit", "num_conventions": 3}, path)
    loaded = load_actor(path, "experiment1", "explicit", "cpu")
    assert loaded.initial_belief(1).shape == (1, 3)
    with pytest.raises(ValueError):
        load_actor(path, "experiment2", "explicit", "cpu")


def test_csv(tmp_path):
    actor, _ = create_models("implicit", 3, torch.device("cpu"))
    _, records = evaluate(actor, lambda _: TinyEnv(), "experiment1", "implicit", episodes=2)
    path = save_records(tmp_path / "evaluation.csv", records)
    assert path.read_text().count("\n") == 3
