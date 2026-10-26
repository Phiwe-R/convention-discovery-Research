# Convention Discovery Under Partner Uncertainty

Refactored shared GridWorld for the Honours AHT research. **The 5×5 grid now uses matrix coordinates (row, column), with (0,0) top-left**, and **eight independently sampled learner goals and eight independently sampled partner goals** replace the original two learner goals and fixed partner goal. Both experiments use the same environment and goal libraries; only the number of available conventions changes.

## Folder layout

```text
convention_discovery_research/
├── environments/
│   ├── conventions.py          # C0--C5, including teacher action generation
│   ├── goal_configurations.py  # Two independent sets of eight goal positions
│   └── gridworld.py            # ONE PettingZoo GridWorld and MAPPO wrapper
├── configs/
│   └── experiments.py          # Experiment 1 uses C0--C2; Experiment 2 uses C0--C5
├── agents/                    # To integrate later
├── training/                  # To integrate later
├── evaluation/                # To integrate later
├── scripts/
│   ├── smoke_test.py
│   └── visualise_conventions.py
├── tests/
│   └── test_gridworld.py
├── results/
├── requirements.txt
└── README.md
```

## Setup in VS Code / Windows PowerShell

Unzip the project; **open the `convention_discovery_research` folder as your VS Code workspace**. In its terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.smoke_test
python -m pytest -q
```

If activation is blocked, run `.venv\Scripts\python.exe` explicitly instead.

## The coordinate convention

All positions are `(row, column)`. `(0,0)` is **top-left**; `(4,4)` is **bottom-right**. All movement and A* distances use toroidal (wraparound) geometry:

| Action | ID | Matrix move |
|---|---|---|
| UP | 0 | `(row-1) % 5, col` |
| DOWN | 1 | `(row+1) % 5, col` |
| LEFT | 2 | `row, (col-1) % 5` |
| RIGHT | 3 | `row, (col+1) % 5` |
| STAY | 4 | no movement |

With the toroidal shortest-path metric, the maximum distance from **start to goal** is **4** (not 8). Visualisation uses exactly these coordinates, displays the position in each cell and does not flip the vertical axis.

## Independently sampled goals

The goal library defines eight learner goals and eight different partner goals.
Each episode selects one from each list **independently**, giving 64 equally likely
combinations. The sets do not overlap, so the goals cannot coincide.

The learner distance is sampled separately from `(1,2,2,3,3,4,4,4)`; this
produces maximum-distance learner starts in 3/8 of episodes without tying
a distance to a particular goal ID. Starts are sampled uniformly from a
convention-independent feasible pool for each (learner goal, partner goal,
distance). Pools are computed lazily and may add initialization overhead.

Optional reset overrides: `learner_goal_id`, `partner_goal_id`,
`target_learner_distance`, and `convention_id`. Goal IDs and target distances
are diagnostic information only; never feed them directly to the learner.

## Conventions (all in one file)

| ID | Behaviour |
|---|---|
| C0 | Travel to hidden learner goal, STAY, then to partner goal |
| C1 | Demonstrate learner's initial A* actions from partner start, then return |
| C2 | Opposite of each initial A* direction, then return |
| C3 | Rotate initial directions clockwise by 90°, then return |
| C4 | Rotate initial directions counterclockwise by 90°, then return |
| C5 | Visit one cell RIGHT of hidden learner goal, STAY, return without visiting goal |

For C1--C5, the partner must never step onto the learner's actual goal. Invalid start combinations are removed *before* sampling, not conditionally rejected for individual conventions.

## How to use the shared GridWorld

```python
from configs.experiments import make_env

env1 = make_env("experiment1")  # C0, C1, C2
env2 = make_env("experiment2")  # C0, C1, C2, C3, C4, C5

obs, info = env1.reset(seed=42)
actor_input = obs["learner"]  # five values: learner_row, learner_col,
                               # partner_row, partner_col, partner_last_action
print("Goal IDs (diagnostic):", info["learner_goal_id"], info["partner_goal_id"])
print("True convention (diagnostic):", info["convention"])
env1.close()
env2.close()
```

To force a specific test case:

```python
env = make_env("experiment2")
obs, info = env.reset(seed=42, options={"convention_id": 5, "learner_goal_id": 7, "partner_goal_id": 2})
```

To visualise one independently selected learner/partner goal combination for all three/six conventions:

```powershell
python -m scripts.visualise_conventions --experiment experiment2 --learner-goal-id 5 --partner-goal-id 2
```

## Observations and rewards: unchanged, **but coordinates are different**

- Actor-observable learner vector: `[learner_row, learner_col, partner_row, partner_col, last_partner_action]` (five integers; sentinel `5` at reset).
- Partner/central-critic observation: `[partner_row, partner_col, learner_row, learner_col, partner_goal_row, partner_goal_col]`.
- Gymnasium wrapper adds ground-truth `convention` only for **loss/evaluation**; do **not** supply it to the actor. Similarly, do not leak the partner's goal through the critic vector to the actor.
- Original navigation rewards remain: `+1` for reaching the learner goal; `-0.02` for invalid learner movement; `-0.01` for a valid nonterminal move; partner receives `0`.
- Episode ends on learner goal arrival or truncates after 30 steps. This is **navigation success**, not proof of convention identification.

### IMPORTANT: existing checkpoints are not compatible

Changing the coordinate semantics, goal distributions, and initial distances substantially changes the MDP. **Do not compare models trained on the older environment as if they were trained under the same conditions.** Retrain both implicit and explicit agents and both experiment configurations using the revised version, with the same evaluation goal distribution and seed protocol.

Even 64 possible goal combinations permit memorisation in principle. For stronger claims about discovery, use held-out goal pairings / start-goal combinations and measure matching accuracy separately from navigation success in the future evaluation refactor.

## Audit: avoid goal-position shortcuts before training

We now sample learner and partner goals independently (64 combinations) and
sample the required learner start-to-goal distance independently. This removes
the fixed pairing and fixed-distance-by-goal shortcuts, but starting positions
can still contain information about the hidden destination.

```powershell
python -m scripts.audit_goal_shortcuts
```

This diagnostic checks learner-goal inference from starting positions for one
fixed partner goal and distance. It is not measured GRU performance or partner
matching. Measure identification separately from navigation success and evaluate
on held-out positions/pairings for stronger generalisation claims.

For a text-only coordinate diagram (no graphics window), try:

```python
from environments.gridworld import SimpleGridworld
world = SimpleGridworld(render_mode="ansi", num_conventions=6)
world.reset(seed=42, options={"learner_goal_id": 5, "partner_goal_id": 2, "convention_id": 0})
print(world.render())
world.close()
```
