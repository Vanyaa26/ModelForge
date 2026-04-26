---
title: ModelForge
emoji: 🔧
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - machine-learning
---

# ModelForge: A Model That Learns to Train Models

**Hugging Face Space:** https://huggingface.co/spaces/vanyatentiwala/modelforge

**GitHub repository:** https://github.com/Vanyaa26/ModelForge.git

**Training notebook:** [`training/AutoLearn_SUBMIT.ipynb`](training/AutoLearn_SUBMIT.ipynb)

**Pitch video:** replace this line with the public YouTube link before final submission.

ModelForge is an OpenEnv environment for training LLM agents to behave like practical ML engineers. On reset, the agent receives a dataset description: feature count, class names, class distribution, sample rows, and a majority-class baseline. On each step, it submits Python training code. The environment executes the code, evaluates the trained model on a held-out test split, and returns real metrics plus a multi-signal reward.

The goal is not just to tune a hyperparameter. The agent has to read the shape of a dataset, choose an appropriate sklearn pipeline, avoid overfitting, recover from failed code, and improve over a baseline.

## Inspiration

Andrej Karpathy's AutoResearch showed a powerful idea: an AI agent can run experiments, measure results, and keep the changes that work. ModelForge starts from that idea but changes the loop in one important way: the agent itself learns. The same Qwen 2.5 1.5B model that writes the training code is updated through SFT and DPO from its own successful and failed attempts.

In short:

```text
env.reset() -> dataset description
agent -> writes training code
env.step(code) -> executes code, returns metrics and reward
SFT/DPO -> update the code-writing model
repeat -> a better ML engineer
```

## Environment

ModelForge follows the OpenEnv server/client pattern with a standard Gym-style API.

- `reset(seed)` chooses one of seven classification datasets and returns an observation.
- `step(ModelforgeAction(code=...))` executes the submitted code against the current train/test split.
- `state` tracks the active episode and step count.
- The server is a FastAPI OpenEnv app configured by `openenv.yaml`.

Datasets:

- `iris`
- `wine`
- `breast_cancer`
- `digits`
- `moons`
- `circles`
- `hard_classify`

## Action

The agent submits Python code:

```python
ModelforgeAction(code="...")
```

The code runs with these variables already available:

- `X_train`, `X_test`
- `y_train`, `y_test`
- `np`

To count as a successful step, the code must train an estimator and assign it to one of:

- `clf`
- `model`
- `pipe`
- `pipeline`
- `estimator`

Example:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, random_state=42)
)
pipe.fit(X_train, y_train)
```

## Observation

The observation gives the agent enough context to reason before writing code:

```python
{
    "dataset_name": "wine",
    "task_type": "classification",
    "num_samples": 178,
    "num_features": 13,
    "num_classes": 3,
    "feature_names": [...],
    "target_names": [...],
    "sample_rows": [...],
    "class_distribution": {...},
    "baseline_accuracy": 0.399,
    "accuracy": 0.0,
    "execution_success": false,
    "reward_breakdown": {}
}
```

After a step, the environment also reports test accuracy, train accuracy, cross-validation stats, overfit gap, execution errors, and whether the submission improved the best model so far.

## Reward

The reward is designed to measure the agent as an ML engineer, not just a one-shot classifier picker.

| Signal | Weight | What it rewards |
|---|---:|---|
| Accuracy | 0.30 | High held-out test accuracy |
| Improvement | 0.30 | Beating the majority-class baseline |
| Efficiency | 0.15 | Reaching improvement in fewer turns |
| Diagnosis | 0.15 | Recovering after crashes or poor attempts |
| No overfit | 0.10 | Keeping train/test gap small |

This makes the environment useful for training behavior like dataset reasoning, error recovery, and model selection.

## Training Evidence

The final notebook uses Hugging Face Transformers, TRL, and PEFT LoRA:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT: train on successful generated training scripts
- DPO: prefer high-accuracy scripts over failed or lower-accuracy scripts
- Evaluation: run fresh environment episodes after each training stage

Observed submit-run trend from `training/AutoLearn_SUBMIT.ipynb`:

| Stage | Success | Avg accuracy | Avg reward |
|---|---:|---:|---:|
| Base Qwen 1.5B | 39/42 | 0.876 | 0.564 |
| After SFT | 13/14 | 0.867 | 0.554 |
| After DPO | 14/14 | 0.853 | 0.588 |

The clearest improvement is reliability and reward: the final DPO model reached a 100% execution success rate on the evaluation episodes and the highest average reward. The transfer check also showed the trained model matching or improving the base model on held-out seeds for circles and wine. The notebook includes the real result table, reward curve, per-dataset comparison, and SFT/DPO loss plot.

## Run Locally

Install dependencies from this environment directory:

```bash
uv sync
```

Run the OpenEnv server:

```bash
uv run server
```

Or run directly:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The deployed server exposes:

- `/web` for the OpenEnv web interface
- `/docs` for API docs
- `/health` for health checks
- `/ws` for WebSocket environment sessions

## Re-run Training

Open the notebook:

```text
training/AutoLearn_SUBMIT.ipynb
```

It installs the needed training dependencies, builds the environment loop, runs base-model episodes, applies SFT, applies DPO, evaluates each stage, and generates the plots used for the submission.

## Deploy

From this directory:

```bash
openenv push --repo-id vanyatentiwala/modelforge
```

Before submitting, verify that the README contains the final public HF Space link and the public video or blog link.
