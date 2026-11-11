# Convention Discovery Under Partner Uncertainty

The repo contains the code, saved models, training logs and evaluation results for the four agents in the report . **Experiment 1** uses 3 partner conventions; **Experiment 2** uses 6. The reported comparisons use training seeds **20, 21 and 22**, with **100,000** and **500,000** training step budgets.

## 1. Agents

| File in `agents/` | Purpose |
|---|---|
| `implicit_gru_agent.py` | **Agent 1:** implicit GRU policy. |
| `goal_guided_actor.py` | **Agent 2:** actor using convention and goal beliefs |
| `joint_belief_agent.py` | **Agent 2:** supporting network for the joint convention-and-goal belief predictions. |
| `explicit_belief_agent.py` | **Agent 3:** GRU-based convention-only belief agent. |
| `goal_only_guided_actor.py` | **Agent 4:** actor using goal beliefs and goal guidance. |
| `centralized_critic.py` | Centralised critic used during MAPPO training. |

## 2. Code folders

| Location | What to find there |
|---|---|
| `configs/experiments.py` | Settings for Experiment 1 (3 conventions) and Experiment 2 (6 conventions). |
| `environments/conventions.py` | Definitions of the partner movement conventions. |
| `environments/goal_configurations.py` | Possible learner and partner goal locations. |
| `environments/gridworld.py` | The 5×5 GridWorld environment and its dynamics. |
| `training/train_mappo.py` | MAPPO training for **Agents 1 and 3**; Agent 3 also uses convention cross-entropy. |
| `scripts/train_goal_guided_batch.py` | Batch-training runner used for **Agent 2**. |
| `scripts/train_goal_only_batch.py` | Batch-training runner used for **Agent 4**. |
| `scripts/visualise_trained_agent.py` | To see the environment for one of the trained Agents (**Agent 4**). |
| `scripts/evaluate_goal_only_batch.py` | Batch-evaluation runner used for **Agent 4**. |
| `evaluation/evaluate_research_batch.py` | Final batch evaluation for **Agent 2** (and supporting implicit comparisons). |
| `tests/` | Development tests; not the final research results. |
| `runs/` | TensorBoard logs; not the primary source of reported figures and tables. |

### Final batch launchers (in the main folder)

| File | Purpose |
|---|---|
| `run_explicit_training.bat` | Runs Agent 2 training through `scripts/train_goal_guided_batch.py`. |
| `run_a3_training.bat` | Runs Agent 4 training through `scripts/train_goal_only_batch.py`. |
| `run_a3_evaluation.bat` | Runs Agent 4 evaluation through `scripts/evaluate_goal_only_batch.py`. |

**Important:** Despite `a3` in the filenames, the two `run_a3_*.bat` files belong to **Agent 4 (goal-only)** in this report. Agents **1 and 3** were trained using `training/train_mappo.py`.

The older direct training entry points `training/pretrain_balanced_belief.py`, `training/pretrain_goal_only.py`, `training/train_goal_guided.py` and `training/train_goal_only.py` are **not the final batch launchers**. **Keep these Python files:** the batch runners call them internally for pretraining and training. `training/train_joint_belief.py` is also a supporting code module. `scripts/plot_balanced_pretraining.py` and the older `evaluation/evaluate_goal_guided.py`, `evaluation/evaluate_goal_only.py` and `evaluation/evaluate_joint_belief.py` are not used for the final report results.

## 3. Results folders

| Location | What to find there |
|---|---|
| `results/batch_training/` | **Agents 1 and 3:** saved models (`.pt`) and training logs for both experiments, all seeds and both budgets. Here `implicit` means Agent 1 and `explicit` means **Agent 3 (convention-only)**. |
| `results/batch_evaluation/` | **Agents 1 and 3:** evaluation results, including `master_results.csv` for goal success. |
| `results/goal_guided_clean/` | **Agent 2:** final training checkpoints, logs and `master_training_curves.csv`. |
| `results/final_evaluation/` | **Agent 2:** final evaluation CSVs for goal success, convention and goal identification, and convention-belief ablation. |
| `results/goal_only/` | **Agent 4:** final training checkpoints and `master_training_curves.csv`. |
| `results/goal_only/evaluation/` | **Agent 4:** `master_evaluation_results.csv`, per-episode results, goal identification and goal-belief ablation. |
| `results/joint_belief/` | Supporting/pretraining outputs; **do not use its older figures as final results**, but retain `balanced_exp1.pt` and `balanced_exp2.pt` because Agent 2's batch runner uses these pretrained belief checkpoints. |

## 4. Where each reported metric comes from

| Metric in Chapter 4 | Source data |
|---|---|
| **Training goal success** (Figures 4.1–4.2) | **Agents 1 & 3:** `results/batch_training/steps_*/logs/`; **Agent 2:** `results/goal_guided_clean/master_training_curves.csv` (`recent_goal_success_pct`); **Agent 4:** `results/goal_only/master_training_curves.csv` (`recent_goal_success_pct`). |
| **PPO policy loss** (Figures 4.3–4.4) | **Agents 1 & 3:** `results/batch_training/steps_*/logs/` (`PPO loss`); **Agent 2:** `results/goal_guided_clean/master_training_curves.csv` (`policy_loss`); **Agent 4:** `results/goal_only/master_training_curves.csv` (`policy_loss`). |
| **Convention and goal belief cross-entropy** (Figure 4.5, Agent 2) | `results/goal_guided_clean/master_training_curves.csv` (`belief_convention_ce` and `belief_goal_ce`). |
| **Evaluation goal success** (Figure 4.6) | **Agents 1 & 3:** `results/batch_evaluation/master_results.csv` (`goal_success_pct`); **Agent 2:** `results/final_evaluation/checkpoint_evaluations.csv` (`goal_success_pct`, `real` mode); **Agent 4:** `results/goal_only/evaluation/master_evaluation_results.csv` (`success_pct`, `real` mode). |
| **Convention identification by timestep** (Figure 4.7, Agent 2) | `results/final_evaluation/explicit_identification_by_timestep.csv` (`convention_id_pct`). |
| **Final goal identification** (Table 4.1) | **Agent 2:** `results/final_evaluation/explicit_identification.csv` (`goal_id_final_pct`); **Agent 4:** `results/goal_only/evaluation/master_evaluation_results.csv` (`final_goal_id_pct`, `real` mode); use the 500,000-step results. |
| **Convention-belief ablation** (Table 4.2, Agent 2) | `results/final_evaluation/checkpoint_evaluations.csv`: compare `real` with `uniform_convention` for the same experiment, seed and budget. |
| **Goal-belief ablation** (Table 4.3, Agent 4) | `results/goal_only/evaluation/master_evaluation_results.csv`: compare `real` with `uniform` for the same experiment, seed and budget. |

Reported summary percentages are averages over seeds **20, 21 and 22**. For individual verification, follow the corresponding experiment/seed/budget subfolders and their `.csv`, `.json`, `.log` and `.pt` files. In the Agent 1/3 batch folders, the label `explicit` refers to Agent 3, **not** Agent 2.
