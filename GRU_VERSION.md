# Independent belief and action GRUs

The explicit agent now uses a 64-unit GRU followed by a linear identity head.
It reads the five learner observation features and carries its hidden state
between timesteps. A separate 64-unit action GRU receives only the current observation and
detached convention probabilities, and outputs actions through a linear head. Cross-entropy alone updates the identifier;
PPO updates the action GRU, action head and critic. Their optimizers remain separate.

`initial_belief` returns uniform probabilities for belief-ablation tests.
`initial_memory` returns `(belief, belief_hidden, action_hidden)` for rollout collection.
The third return from `forward`, and fourth from `get_action` and
`evaluate_actions`, is this complete memory tuple. Reset it each episode.

| Inference parameters | Three conventions | Six conventions |
| --- | ---: | ---: |
| Belief GRU and identity head | 13,827 | 14,022 |
| Action GRU and action head | 14,533 | 15,109 |
| Total explicit model | 28,360 | 29,131 |
| Unchanged implicit model | 18,117 | 18,117 |

Train a fresh explicit model from the project root:

```bash
python -m training.train_mappo --experiment experiment1 --agent explicit --timesteps 500000 --seed 20 --output-dir results_gru
```

For six conventions, use `--experiment experiment2`. Existing MLP and belief-GRU-only explicit
checkpoints are incompatible. Implicit checkpoint compatibility is unchanged.
Evaluate the new checkpoint explicitly:

```bash
python -m evaluation.evaluate_mappo --experiment experiment1 --agent explicit --checkpoint results_gru/experiment1/explicit/seed_20.pt
```

Collection, evaluation, belief sensitivity checks and relevant tests have been
adapted for recurrent belief memory. Environment dynamics, rewards, implicit
agent, critic and PPO settings are unchanged. The two GRUs have independent
hidden states and optimizers. Uniform-belief evaluation carries action memory
across steps; sensitivity comparisons share the same pre-step action memory. This version adds
memory; it does not introduce a multiplicative Bayesian update, pretraining,
or a fixed identification horizon. The older README describes the original
MLP architecture; this note describes the two-GRU variant.
