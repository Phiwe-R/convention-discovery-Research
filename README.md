# Convention Discovery Under Partner Uncertainty

### Implicit Memory vs Explicit Belief Tracking in Ad Hoc Teamwork

**Research Area:** Multi-Agent Reinforcement Learning (MARL), Ad Hoc Teamwork, Convention Discovery  
**Institution:** University of the Witwatersrand  
**Programme:** BSc Honours in Computational and Applied Mathematics  
**Status:** Training and initial evaluation completed; results analysis in progress

---

## 1. Project Overview

In ad hoc teamwork, an agent must cooperate with unfamiliar partners without relying on prior coordination or communication.

One challenge is **partner uncertainty**: the learner may know which conventions are possible but does not know which convention the current partner follows.

This project investigates how agents learn to recognise and adapt to different partner conventions.

We compare two approaches:

- **Implicit learning:** A recurrent neural network uses memory of previous observations to select actions without explicitly representing the partner's identity.
- **Explicit belief tracking:** A neural network maintains a probability distribution over possible partner conventions and supplies that information to the action-selection network.

Both approaches are trained using a Multi-Agent Proximal Policy Optimization (MAPPO)-based framework.

### Research Question

> Under what conditions does convention discovery in ad hoc teamwork transition from being achievable through implicit memory to requiring explicit belief tracking?

### Research Objectives

1. Design a controlled GridWorld environment containing deterministic partner conventions.
2. Implement implicit and explicit learning architectures.
3. Compare learning performance under different levels of convention uncertainty.
4. Evaluate both goal-reaching performance and convention-identification accuracy.
5. Investigate whether explicit belief representations influence action selection.
6. Assess performance across multiple training seeds and training budgets.

---

## 2. Environment

The environment is a **5 × 5 toroidal GridWorld**, meaning that agents can wrap around grid boundaries.

It contains two agents:

| Agent | Role |
|---|---|
| Learner | Observes the partner and attempts to reach its assigned goal |
| Partner | Follows a deterministic convention-dependent policy |

### Environment Properties

| Property | Configuration |
|---|---|
| Grid size | 5 × 5 |
| Movement | Toroidal (wraparound) |
| Number of agents | 2 |
| Action space | UP, DOWN, LEFT, RIGHT, STAY |
| Maximum episode length | 12 steps |
| Learner observation dimension | 5 |
| Number of learner-goal configurations | 8 |
| Number of partner-goal configurations | 8 |
| Partner behaviour | Deterministic, convention-dependent |
| Active convention | Hidden from the learner |

The learner's observation includes its position, the partner's position, and the partner's previous action.

The active partner convention is sampled at the beginning of an episode and is not directly provided to the learner.

### Goal Configurations

The environment varies both learner and partner goals between episodes.

The goal configurations are defined in:

`environments/goal_configurations.py`

### Partner Conventions

The environment supports the following conventions:

| Convention | Behaviour |
|---|---|
| C0 | Goal visiting |
| C1 | Route mirroring |
| C2 | Inverse mirroring |
| C3 | Clockwise route rotation |
| C4 | Counterclockwise route rotation |
| C5 | Waypoint signalling |

The partner follows a deterministic movement policy based on its active convention and the environment configuration.

---

## 3. Agent Architectures

### 3.1 Implicit Agent — GRU-MAPPO

The implicit agent uses a Gated Recurrent Unit (GRU) to maintain information about previous observations.

**Architecture**

```text
Observation at time t
        |
        v
   GRU Network
   Hidden size: 64
        |
        v
   Hidden State
        |
        v
   Policy Head
        |
        v
 Action Probabilities
        |
        v
    Selected Action
```

The GRU hidden state serves as an implicit memory of the interaction history.

The agent is not explicitly supervised to classify the active convention.

#### Convention Identification Using a Probe

After training, the GRU is frozen and a separate linear classifier is trained to predict the active convention from its hidden states.

```text
Frozen GRU
    |
    v
Hidden State (64)
    |
    v
Linear Probe
    |
    v
Predicted Convention
```

The probe is used only for representation analysis; it is not part of the original action-selection policy.

### 3.2 Explicit Agent — Belief-MAPPO

The explicit agent contains two cooperating neural networks:

1. A belief network.
2. An action-selection network.

**Architecture**

```text
Previous Belief ---------+
                         |
Current Observation -----+
                         |
                         v
                  Belief Network
                         |
                         v
                  Updated Belief
                         |
              +----------+
              |
Current Observation
              |
              v
           Policy MLP
              |
              v
      Action Probabilities
              |
              v
         Selected Action
```

The belief network outputs a probability distribution over possible partner conventions.

The network is trained using a cross-entropy objective with the true convention labels.

The policy is trained using the MAPPO/PPO objective.

### Centralized Training

Both approaches use a centralized critic during training.

The critic has access to privileged environment information for value estimation. This information is not directly supplied to the learner's action-selection policy.

---

## 4. Experimental Design

Two main experiments investigate convention uncertainty.

| Experiment | Conventions | Convention Count |
|---|---|---:|
| Experiment 1 | C0, C1, C2 | 3 |
| Experiment 2 | C0, C1, C2, C3, C4, C5 | 6 |

Experiment 1 uses a smaller convention library, while Experiment 2 uses a larger library.

Because the convention sets differ, results should not be interpreted as isolating convention count from every other aspect of task difficulty.

### Training Configurations

| Parameter | Value |
|---|---|
| Algorithm | MAPPO/PPO |
| Learning rate | 0.00025 |
| Discount factor | 0.99 |
| GAE lambda | 0.95 |
| PPO update epochs | 4 |
| Training seeds | 20, 21, 22 |
| Training budgets | 100,000 and 500,000 timesteps |
| Total models | 24 |
| Evaluation episodes | 1,000 per model |

The 100k and 500k models were trained independently from scratch.

---

## 5. Evaluation Metrics

Performance is measured along two dimensions: navigation and convention identification.

### Navigation Performance

- Goal success rate
- Average episode reward
- Average episode length
- Episode truncation rate

### Convention Identification

- Final-step identification accuracy
- Identification accuracy across timesteps
- Linear-probe classification accuracy for the implicit agent
- Explicit belief-network classification accuracy

Chance identification accuracy is:

- **33.33%** for Experiment 1.
- **16.67%** for Experiment 2.

### Belief Dependence

For the explicit agent, evaluation compares:

- Navigation using the learned belief distribution.
- Navigation using a uniform belief distribution.

This tests whether learned beliefs contribute to action-selection performance.

---

## 6. Repository Structure

```text
convention-discovery-Research/
│
├── agents/
│   └── Neural network architectures
│
├── environments/
│   ├── gridworld.py
│   └── goal_configurations.py
│
├── configs/
│   └── Experiment configurations
│
├── training/
│   └── train_mappo.py
│
├── evaluation/
│   ├── evaluate_mappo.py
│   ├── run_all_evaluations.py
│   └── train_implicit_probe.py
│
├── scripts/
│   └── run_all_trainings.py
│
├── results/
│   ├── batch_training/
│   └── batch_evaluation/
│
├── README.md
├── requirements.txt
├── CONTRIBUTING.md
└── .gitignore
```

The repository contains implementation code, experiment configurations, evaluation outputs, and trained model checkpoints where available.

---

## 7. Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Phiwe-R/convention-discovery-Research.git
cd convention-discovery-Research
```

If checkpoints are managed through Git LFS, install Git LFS before cloning or run `git lfs pull` after cloning.

### Step 2 — Create a Python environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
python -m pip install -r requirements.txt
```

Run commands from the project root so Python can resolve the package imports.

---

## 8. Training Models

The project supports automated training across experiments, agents, seeds and training budgets.

### Preview the training jobs

```bash
python -m scripts.run_all_trainings --dry-run
```

### Train all research configurations

```bash
python -m scripts.run_all_trainings --seeds 20 21 22 --timesteps 100000 500000
```

### Train one model configuration

```bash
python -m scripts.run_all_trainings --seeds 20 --timesteps 100000 --experiments experiment1 --agents implicit
```

Completed checkpoints are skipped automatically unless forced to retrain.

Model checkpoints are stored under:

`results/batch_training/`

---

## 9. Evaluating Models

### Preview the evaluation jobs

```bash
python -m evaluation.run_all_evaluations --dry-run
```

### Evaluate all models

```bash
python -m evaluation.run_all_evaluations
```

The evaluation pipeline:

1. Loads saved checkpoints.
2. Evaluates navigation on held-out episodes.
3. Measures convention-identification performance.
4. Trains and tests frozen-GRU linear probes for implicit agents.
5. Compares learned and uniform beliefs for explicit agents.
6. Saves individual evaluation files and combined summaries.

### Evaluation Outputs

The main summary is saved to:

`results/batch_evaluation/master_results.csv`

Each model also has separate evaluation records and summary files.

---

## 10. Research Results

Initial evaluation across three training seeds shows that navigation success generally increases with training budget.

Convention-identification accuracy, however, remains relatively close to chance in both experiments.

In addition, replacing learned beliefs with uniform beliefs has shown relatively small effects on explicit-agent goal success.

These observations motivate further investigation into whether successful navigation necessarily requires accurate partner identification.

**Important:** Goal success and convention-identification accuracy measure different behaviours. Strong navigation performance alone is not evidence of successful convention discovery.

Detailed evaluation CSVs and training logs are available in the `results/` directory.

---

## 11. Current Research Status

**Completed**

- GridWorld implementation
- Deterministic partner conventions
- Implicit GRU architecture
- Explicit belief architecture
- MAPPO training pipeline
- Two experimental configurations
- Multi-seed training
- Evaluation of 24 models
- Automated identification probing
- Uniform-belief evaluation

**Ongoing**

- Statistical analysis across seeds
- Research visualisations
- Training stability analysis
- Additional GRU representation diagnostics
- Research report preparation

---

## 12. Collaboration and Contributions

This repository is intended to support research collaboration and reproducibility.

Potential contribution areas include:

- Agent architecture improvements
- Belief representation and calibration
- Alternative probing methods
- Experimental controls and robustness tests
- Training diagnostics
- Statistical evaluation
- Environment validation
- Research documentation

Contributors should preserve the original experimental configurations when making comparisons.

New experimental variants should be documented separately, including changes to rewards, goal configurations, network architectures and hyperparameters.

Please see `CONTRIBUTING.md` for contribution guidance.

---

## 13. Reproducibility Notes

- Use fixed random seeds when comparing models.
- Use the same evaluation episodes for corresponding agents.
- Report performance across multiple training seeds.
- Keep probe training and testing data separate.
- Do not update the implicit GRU when training diagnostic probes.
- Distinguish navigation improvements from convention-identification improvements.
- Document environment or reward changes before running new experiments.

---

## 14. License and Citation

A software license and formal research citation will be added after the applicable research-sharing requirements have been confirmed.

---

**Research Project:** Convention Discovery Under Partner Uncertainty  
**University:** University of the Witwatersrand  
**Field:** Computational and Applied Mathematics / Multi-Agent Reinforcement Learning
