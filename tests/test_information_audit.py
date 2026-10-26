"""Regression safeguards for observation information and controlled identifiability."""
import numpy as np
import pytest
from configs.experiments import make_env


@pytest.mark.parametrize('experiment,n', [('experiment1', 3), ('experiment2', 6)])
def test_actor_input_excludes_privileged_labels(experiment, n):
    env = make_env(experiment)
    try:
        for convention in range(n):
            obs, info = env.reset(seed=23, options={
                'learner_goal_id': 5, 'partner_goal_id': 2,
                'target_learner_distance': 3, 'convention_id': convention})
            learner = obs['learner']
            assert learner.shape == (5,)
            assert int(obs['convention'][0]) == convention  # supervision ONLY
            assert int(info['convention']) == convention
            assert tuple(learner[:2]) == tuple(env.env.learner_pos)
            assert tuple(learner[2:4]) == tuple(env.env.partner_pos)
            assert int(learner[4]) == 5  # no partner action yet
            # No privileged labels among five learner observation values.
            before = learner.copy()
            assert len(before) == 5
    finally:
        env.close()


@pytest.mark.parametrize('experiment,n', [('experiment1', 3), ('experiment2', 6)])
def test_starts_independent_of_convention(experiment, n):
    start_pairs = set()
    env = make_env(experiment)
    try:
        for c in range(n):
            env.reset(seed=17, options={'learner_goal_id': 5,
                'partner_goal_id': 2, 'target_learner_distance': 4,
                'convention_id': c})
            start_pairs.add((tuple(env.env.learner_start),tuple(env.env.partner_start)))
        assert len(start_pairs) == 1
    finally:
        env.close()


def test_no_initial_action_leak_for_different_conventions():
    env = make_env('experiment2')
    observations = []
    try:
        for tau in range(6):
            obs,_ = env.reset(seed=19,options={'learner_goal_id':5,
                'partner_goal_id':2,'target_learner_distance':4,'convention_id':tau})
            observations.append(tuple(obs['learner'].tolist()))
        assert len(set(observations)) == 1
    finally:
        env.close()
