"""Regression tests for independent goals and matrix-style conventions."""
import numpy as np
import pytest
from configs.experiments import make_env
from environments.gridworld import SimpleGridworld
from environments.goal_configurations import LEARNER_GOALS, PARTNER_GOALS, DISTANCE_OPTIONS, validate_goal_configurations


def test_goal_libraries():
    validate_goal_configurations()
    assert len(set(LEARNER_GOALS)) == 8
    assert len(set(PARTNER_GOALS)) == 8
    assert not set(LEARNER_GOALS) & set(PARTNER_GOALS)
    assert DISTANCE_OPTIONS.count(4) >= 3


def test_matrix_movement():
    env = SimpleGridworld()
    assert env._move((0, 0), 0).tolist() == [4, 0]
    assert env._move((0, 0), 1).tolist() == [1, 0]
    assert env._move((0, 0), 2).tolist() == [0, 4]
    assert env._move((0, 0), 3).tolist() == [0, 1]
    assert env._toroidal_heuristic((2, 2), (0, 0)) == 4
    env.close()


@pytest.mark.parametrize('experiment,count', [('experiment1',3),('experiment2',6)])
def test_experiment_configuration(experiment,count):
    env = make_env(experiment)
    try:
        assert len(env.env.conventions) == count
        assert len(env.env.learner_goals) == 8
        assert len(env.env.partner_goals) == 8
    finally:
        env.close()


@pytest.mark.parametrize('lg,pg', [(0,0),(0,7),(7,0),(7,7),(5,2)])
def test_independent_goals_and_all_conventions(lg,pg):
    env=SimpleGridworld(num_conventions=6)
    try:
        for c in range(6):
            options={'learner_goal_id':lg,'partner_goal_id':pg,'target_learner_distance':4,'convention_id':c}
            obs, info = env.reset(seed=42,options=options)
            assert tuple(env.learner_goal)==LEARNER_GOALS[lg]
            assert tuple(env.partner_goal)==PARTNER_GOALS[pg]
            assert env._toroidal_heuristic(env.learner_start,env.learner_goal)==4
            assert info['learner']['learner_goal_id']==lg
            assert info['learner']['partner_goal_id']==pg
            assert not np.array_equal(env.learner_pos,env.partner_pos)
            assert env.observation_space('learner').contains(obs['learner'])
            pos=env.partner_start.copy()
            for a in env.partner_actions:
                pos=env._move(pos,a)
            assert np.array_equal(pos,env.partner_goal)
    finally:
        env.close()


def test_sampling_does_not_depend_on_active_convention():
    env=SimpleGridworld(num_conventions=6)
    try:
        starts=set()
        for c in range(6):
            env.reset(seed=21, options={'learner_goal_id':3,'partner_goal_id':5,'target_learner_distance':4,'convention_id':c})
            starts.add((tuple(env.learner_start),tuple(env.partner_start)))
        assert len(starts)==1
    finally:
        env.close()


def test_independent_random_goal_selection():
    env=SimpleGridworld(num_conventions=6)
    try:
        selected=set()
        for _ in range(100):
            env.reset()
            selected.add((env.active_learner_goal_id,env.active_partner_goal_id))
        assert len({x for x,_ in selected})==8
        assert len({y for _,y in selected})==8
        assert len(selected)>30
    finally:
        env.close()


def test_wrapper_ids_are_diagnostic_only():
    env=make_env('experiment2')
    try:
        obs,info=env.reset(seed=10,options={'learner_goal_id':3,'partner_goal_id':6,'convention_id':5})
        assert set(obs)=={'learner','partner','convention'}
        assert obs['learner'].shape==(5,)
        assert info['learner_goal_id']==3
        assert info['partner_goal_id']==6
        assert 'learner_goal_id' not in obs
        assert env.observation_space.contains(obs)
    finally:
        env.close()
