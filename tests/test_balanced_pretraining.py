"""Tests for labelled, balanced GRU pretraining. Run with pytest."""
from collections import Counter

import torch

from agents.joint_belief_agent import JointBeliefActor
from training.pretrain_balanced_belief import (
    balanced_plan, generate_balanced_dataset, collate_batch,
    BeliefWithMovement, batch_stats, timestep_metrics,
    SPLIT_SEED_START,
)
from torch.utils.data import DataLoader
from training.pretrain_balanced_belief import ObservationEpisodes


def test_plan_is_balanced_across_conventions_and_goal_pairs():
    plan = balanced_plan(6, 65, seed=12)
    counts = Counter(c for c, lg, pg in plan)
    assert [counts[c] for c in range(6)] == [65] * 6
    for c in range(6):
        pair_counts = Counter((lg, pg) for ct, lg, pg in plan if ct == c)
        assert len(pair_counts) == 64
        assert set(pair_counts.values()) <= {1, 2}
    assert len(set(SPLIT_SEED_START.values())) == 3


def test_data_labels_and_actions_are_valid():
    rows = generate_balanced_dataset('experiment2', 2, 'train', seed=3)
    assert Counter(label for obs, label, goal, acts in rows) == {c: 2 for c in range(6)}
    for obs, label, goal, acts in rows:
        assert obs.ndim == 2 and obs.shape[1] == 5
        assert 1 <= obs.shape[0] <= 12
        assert len(acts) == len(obs)
        assert 0 <= label < 6 and 0 <= goal < 8
        assert (acts >= 0).all() and (acts < 5).all()
        assert obs[0, 4] == 5  # partner action unknown at reset


def test_gradients_are_only_in_belief_and_movement_networks():
    rows = generate_balanced_dataset('experiment2', 2, 'val', seed=3)
    net = BeliefWithMovement(6)
    batch = collate_batch(rows)
    result = batch_stats(net, batch, 'cpu')
    assert result['loss'].isfinite()
    result['loss'].backward()
    assert net.actor.belief_gru.weight_ih_l0.grad is not None
    assert net.actor.convention_head.weight.grad is not None
    assert net.actor.goal_head.weight.grad is not None
    assert net.movement_head.weight.grad is not None
    assert net.actor.actor_gru.weight_ih_l0.grad is None
    assert net.actor.action_head.weight.grad is None


def test_saved_pretraining_weights_fit_actor_without_policy_leakage(tmp_path):
    model = BeliefWithMovement(6)
    checkpoint = tmp_path / 'pre.pt'
    torch.save({
        'experiment':'experiment2',
        'belief_gru':model.actor.belief_gru.state_dict(),
        'convention_head':model.actor.convention_head.state_dict(),
        'goal_head':model.actor.goal_head.state_dict(),
        'movement_head':model.movement_head.state_dict(),
    }, checkpoint)
    loaded = torch.load(checkpoint, weights_only=True)
    policy = JointBeliefActor(num_conventions=6)
    policy.belief_gru.load_state_dict(loaded['belief_gru'])
    policy.convention_head.load_state_dict(loaded['convention_head'])
    policy.goal_head.load_state_dict(loaded['goal_head'])
    assert torch.equal(policy.belief_gru.weight_ih_l0, model.actor.belief_gru.weight_ih_l0)
    # PPO log probabilities cannot send gradients into these belief parameters.
    x = torch.randn(2, 4, 5)
    actions = torch.randint(0, 5, (2, 4))
    logp, entropy, c, g = policy.evaluate_actions(x, actions)
    (logp.mean() - 0.01 * entropy.mean()).backward()
    assert policy.belief_gru.weight_ih_l0.grad is None
    assert policy.actor_gru.weight_ih_l0.grad is not None


def test_timestep_metrics_have_real_denominators():
    rows = generate_balanced_dataset('experiment2', 3, 'test', seed=2)
    loader = DataLoader(ObservationEpisodes(rows), batch_size=8,
                        collate_fn=collate_batch)
    result = timestep_metrics(BeliefWithMovement(6), loader, 'cpu')
    assert result[0]['timestep'] == 1
    assert result[0]['episodes'] == len(rows)
    assert all(0 <= row['convention_accuracy_pct'] <= 100 for row in result)
    assert all(0 <= row['goal_accuracy_pct'] <= 100 for row in result)
    assert all(0 <= row['goal_true_probability'] <= 1 for row in result)
