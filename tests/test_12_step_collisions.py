"""Focused collision, termination, and horizon regression tests.

Run from project root: python -m pytest tests/test_12_step_collisions.py -q
All coordinates are (row, column) on a 5x5 torus.
"""
import numpy as np
import pytest
from environments.gridworld import SimpleGridworld


def staged(learner=(2, 2), partner=(2, 3), sequence=(2,), goal=(4, 4), step=0):
    """A controlled test state; not a sampled policy episode."""
    env = SimpleGridworld(num_conventions=6)
    env.reset(seed=42, options={"learner_goal_id": 0, "partner_goal_id": 0,
                                "target_learner_distance": 2, "convention_id": 0})
    env.max_steps = 12
    env.learner_pos = np.array(learner, dtype=np.int32)
    env.partner_pos = np.array(partner, dtype=np.int32)
    env.learner_goal = np.array(goal, dtype=np.int32)
    env.partner_actions = list(sequence)
    env.convention_step = 0
    env.current_step = step
    env.partner_trajectory = [env.partner_pos.copy()]
    env.agents = env.possible_agents.copy()
    return env


def run(env, action):
    out = env.step({"learner": action})
    assert not np.array_equal(env.learner_pos, env.partner_pos), "agents overlap"
    return out


def test_same_target_partner_priority():
    # Partner (2,3) LEFT -> (2,2); learner (1,2) DOWN -> (2,2)
    env = staged(learner=(1, 2), partner=(2, 3), sequence=(2,))
    _, rewards, _, _, _ = run(env, 1)
    assert tuple(env.partner_pos) == (2, 2)
    assert tuple(env.learner_pos) == (1, 2)
    assert env.convention_step == 1
    assert rewards["learner"] == -0.02
    env.close()


def test_swap_both_wait_and_retry():
    # Partner LEFT would swap with learner moving RIGHT.
    env = staged(learner=(2, 2), partner=(2, 3), sequence=(2,))
    _, rewards, _, _, _ = run(env, 3)
    assert tuple(env.partner_pos) == (2, 3)
    assert tuple(env.learner_pos) == (2, 2)
    assert env.convention_step == 0
    assert rewards["learner"] == -0.02
    # Learner subsequently moves UP: partner can complete the retried LEFT.
    run(env, 0)
    assert tuple(env.partner_pos) == (2, 2)
    assert tuple(env.learner_pos) == (1, 2)
    assert env.convention_step == 1
    env.close()


def test_partner_blocked_when_learner_stays_then_retries():
    env = staged(learner=(2, 2), partner=(2, 3), sequence=(2,))
    _, rewards, _, _, _ = run(env, 4)
    assert tuple(env.learner_pos) == (2, 2)
    assert tuple(env.partner_pos) == (2, 3)
    assert env.convention_step == 0
    assert rewards["learner"] == -0.01
    run(env, 0)
    assert tuple(env.partner_pos) == (2, 2)
    assert env.convention_step == 1
    env.close()


def test_reserved_future_cell_blocks_learner():
    # Partner (2,3) RIGHT -> (2,4), then DOWN reserves (3,4).
    # Learner (3,3) RIGHT attempts reserved (3,4).
    env = staged(learner=(3, 3), partner=(2, 3), sequence=(3, 1))
    _, rewards, _, _, _ = run(env, 3)
    assert tuple(env.partner_pos) == (2, 4)
    assert tuple(env.learner_pos) == (3, 3)
    assert env.convention_step == 1
    assert rewards["learner"] == -0.02
    env.close()


def test_learner_cannot_enter_stationary_partner_cell():
    env = staged(learner=(2, 2), partner=(2, 3), sequence=())
    _, rewards, _, _, _ = run(env, 3)
    assert tuple(env.partner_pos) == (2, 3)
    assert tuple(env.learner_pos) == (2, 2)
    assert rewards["learner"] == -0.02
    env.close()


def test_wraparound_collision():
    # Partner LEFT from column zero wraps to column four.
    env = staged(learner=(0, 4), partner=(0, 0), sequence=(2,))
    run(env, 4)
    assert tuple(env.partner_pos) == (0, 0)
    assert env.convention_step == 0
    env.close()


def test_truncates_exactly_at_step_12():
    env = staged(learner=(2, 2), partner=(4, 0), sequence=(), step=0)
    for index in range(12):
        _, _, terms, truncs, infos = run(env, 4)
        assert infos["learner"]["episode_step"] == index + 1
        assert not terms["learner"]
        assert truncs["learner"] == (index == 11)
    assert env.agents == []
    with pytest.raises(RuntimeError):
        env.step({"learner": 4})
    env.close()


def test_success_before_horizon_terminates_not_truncates():
    env = staged(learner=(2, 2), partner=(4, 0), sequence=(), goal=(2, 3))
    _, rewards, terms, truncs, info = run(env, 3)
    assert rewards["learner"] == 1.0
    assert terms["learner"] and not truncs["learner"]
    assert info["learner"]["task_success"] is True
    assert env.agents == []
    env.close()


def test_success_on_last_step_is_termination_not_truncation():
    env = staged(learner=(2, 2), partner=(4, 0), sequence=(), goal=(2, 3), step=11)
    _, rewards, terms, truncs, info = run(env, 3)
    assert rewards["learner"] == 1.0
    assert info["learner"]["episode_step"] == 12
    assert terms["learner"] is True
    assert truncs["learner"] is False
    env.close()


@pytest.mark.parametrize("convention_id", range(6))
def test_real_episode_never_exceeds_12_steps(convention_id):
    env = SimpleGridworld(num_conventions=6)
    env.max_steps = 12
    try:
        for seed in (3, 17, 29):
            env.reset(seed=seed, options={"convention_id": convention_id})
            for i in range(12):
                if not env.agents:
                    break
                _, _, term, trunc, info = run(env, 4)
                if term["learner"] or trunc["learner"]:
                    assert i < 12
                    break
            assert not env.agents
            assert env.current_step <= 12
    finally:
        env.close()


def test_all_goal_pairs_and_conventions_under_12_step_horizon():
    """Stress-test 64 independent goal pairs x 6 conventions with scripted STAY.

    Completion of the partner's path is intentionally NOT required: the learner
    can block it, but the environment must stay safe and finish by step 12.
    """
    env = SimpleGridworld(num_conventions=6)
    env.max_steps = 12
    try:
        for lg in range(8):
            for pg in range(8):
                for c in range(6):
                    env.reset(seed=7, options={"learner_goal_id": lg,
                            "partner_goal_id": pg, "convention_id": c,
                            "target_learner_distance": 4})
                    for _ in range(12):
                        if not env.agents:
                            break
                        run(env, 4)
                    assert not env.agents
                    assert 1 <= env.current_step <= 12
    finally:
        env.close()
