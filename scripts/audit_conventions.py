"""Audit convention identifiability and observation/metadata leakage.

Run from project root: python -m scripts.audit_conventions
Does NOT train an agent; uses identical scripted STAY actions for every convention.
"""
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from configs.experiments import make_env
from environments.goal_configurations import LEARNER_GOALS, PARTNER_GOALS


def unwrap(env):
    return env.env if hasattr(env, 'env') else env


def audit(experiment='experiment2', seeds=(7, 13), horizon=12):
    n = 3 if experiment == 'experiment1' else 6
    # Hold BOTH hidden goals and starts constant when comparing conventions.
    counts = Counter()
    pairs = Counter()
    exposed_label = False
    leaked_goal = False
    metadata_keys = set()
    total = 0
    for lg_id in range(len(LEARNER_GOALS)):
        for pg_id in range(len(PARTNER_GOALS)):
            for seed in seeds:
                histories = []
                starts = []
                for tau in range(n):
                    env = make_env(experiment, debug=False)
                    raw = unwrap(env)
                    try:
                        obs, info = env.reset(seed=seed, options={
                            'learner_goal_id': lg_id,
                            'partner_goal_id': pg_id,
                            'convention_id': tau,
                            'target_learner_distance': 4,
                        })
                        # Wrapper observation is a dict with privileged fields.
                        # Verify actor observation is exactly the five-feature learner vector.
                        learner = obs['learner']
                        assert learner.shape == (5,), learner.shape
                        assert tuple(learner[:2]) == tuple(raw.learner_pos)
                        assert tuple(learner[2:4]) == tuple(raw.partner_pos)
                        assert learner[4] == 5
                        exposed_label |= ('convention' in obs)
                        leaked_goal |= any('goal' in k for k in obs.keys() if k != 'learner')
                        metadata_keys.update(info.keys())
                        starts.append((tuple(raw.learner_start), tuple(raw.partner_start)))
                        hist = [tuple(learner.tolist())]
                        for t in range(horizon):
                            if not raw.agents:
                                hist.append(('ended',))
                                continue
                            obs, _, term, trunc, info = env.step(4)
                            hist.append(tuple(obs['learner'].tolist()))
                        histories.append(tuple(hist))
                    finally:
                        env.close()
                assert len(set(starts)) == 1, (experiment, lg_id, pg_id, seed, starts)
                total += 1
                for t in (0, 1, 2, 3, 4, 6, 8, 12):
                    groups = defaultdict(list)
                    for tau, hist in enumerate(histories):
                        groups[hist[:t+1]].append(tau)
                    counts[(t,'unambiguous')] += sum(len(v)==1 for v in groups.values())
                    counts[(t,'possible')] += n
                    counts[(t,'fully_identifiable')] += int(len(groups)==n)
                for a,b in combinations(range(n),2):
                    first = next((t for t in range(1,horizon+1)
                                  if histories[a][:t+1]!=histories[b][:t+1]), None)
                    if first is None: pairs[(a,b,'never')] += 1
                    else: pairs[(a,b,first)] += 1
    print(f'\n{experiment}: {total} matched scenarios, {n} conventions each, horizon={horizon}')
    print('Note: identifiability is conditional on fixed goals/starts and scripted learner STAY.')
    print('Step | individually distinguishable types | scenarios with ALL types distinct')
    for t in (0,1,2,3,4,6,8,12):
        print(f'{t:>4} | {counts[(t,"unambiguous")]/counts[(t,"possible")]:>7.1%}'
              f'                         | {counts[(t,"fully_identifiable")]/total:>7.1%}')
    print('Convention pairs still indistinguishable through step 12:')
    for a,b in combinations(range(n),2):
        print(f'  C{a}/C{b}: {pairs[(a,b,"never")]}/{total}')
    print('Reset learner observation does not directly contain convention or goal IDs.')
    if exposed_label:
        print('WARNING: Wrapper observation ALSO contains ground-truth convention; ACTOR MUST ONLY USE obs["learner"].')
    if leaked_goal:
        print('WARNING: Wrapper observation ALSO contains privileged fields; restrict actor inputs.')
    print(f'Reset info fields: {sorted(metadata_keys)} (evaluation labels only)')
    return counts


if __name__ == '__main__':
    audit('experiment1')
    audit('experiment2')
