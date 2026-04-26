"""AutoLearn Environment — where LLM agents learn to manage ML training.

Each episode: agent gets a dataset, makes sequential training decisions,
observes REAL metrics (loss, per-class accuracy, overfitting gap),
and gets rewarded on final model quality.
"""

from __future__ import annotations

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ModelforgeAction, ModelforgeObservation
    from ..engine.datasets import get_dataset, list_datasets, DatasetInfo
    from ..engine.trainer import TrainingSession
    from ..engine.reward import compute_rewards
except ImportError:
    from models import ModelforgeAction, ModelforgeObservation
    from engine.datasets import get_dataset, list_datasets, DatasetInfo
    from engine.trainer import TrainingSession
    from engine.reward import compute_rewards


class ModelforgeEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._dataset_names = list_datasets()
        self._current_dataset: DatasetInfo | None = None
        self._session: TrainingSession | None = None
        self._episode_count = 0
        self._rng = random.Random(42)
        self._budget_turns = 8

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> ModelforgeObservation:
        self._episode_count += 1
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        if seed is not None:
            self._rng = random.Random(seed)

        dataset_name = self._rng.choice(self._dataset_names)
        self._current_dataset = get_dataset(dataset_name, seed=seed or self._rng.randint(0, 10000))

        self._session = TrainingSession(
            X_train=self._current_dataset.X_train,
            X_test=self._current_dataset.X_test,
            y_train=self._current_dataset.y_train,
            y_test=self._current_dataset.y_test,
            budget_turns=self._budget_turns,
        )

        return self._make_observation(done=False, reward=0.0, metrics={})

    def step(self, action: ModelforgeAction, timeout_s: float | None = None, **kwargs) -> ModelforgeObservation:
        self._state.step_count += 1

        if self._current_dataset is None or self._session is None:
            return ModelforgeObservation(
                error_message="No active session. Call reset() first.",
                done=True, reward=0.0,
            )

        metrics = self._session.execute_and_evaluate(action.code)

        budget_exhausted = metrics["budget_remaining"] <= 0
        agent_stopped = "stop" in action.code.lower().strip()[:20]
        done = budget_exhausted or agent_stopped

        if done:
            summary = self._session.get_summary()
            rewards = compute_rewards(summary, self._current_dataset.baseline_accuracy)
            return self._make_observation(
                done=True,
                reward=rewards.total,
                metrics=metrics,
                reward_breakdown=rewards.to_dict(),
                session_summary=summary,
            )

        return self._make_observation(done=False, reward=0.0, metrics=metrics)

    def _make_observation(self, done, reward, metrics, reward_breakdown=None, session_summary=None):
        ds = self._current_dataset
        if ds is None:
            return ModelforgeObservation(done=done, reward=reward)

        return ModelforgeObservation(
            dataset_name=ds.name,
            task_type=ds.task_type,
            num_samples=ds.num_samples,
            num_features=ds.num_features,
            num_classes=ds.num_classes,
            feature_names=ds.feature_names,
            target_names=ds.target_names,
            sample_rows=ds.sample_rows,
            class_distribution=ds.class_distribution,
            baseline_accuracy=ds.baseline_accuracy,
            accuracy=metrics.get("accuracy", 0.0),
            execution_success=metrics.get("success", False),
            error_message=metrics.get("error", "") or "",
            train_time=metrics.get("train_time", 0.0),
            reward_breakdown=reward_breakdown or {},
            episode_number=self._episode_count,
            step_number=self._state.step_count,
            done=done,
            reward=reward,
            metadata={
                "train_accuracy": metrics.get("train_accuracy", 0.0),
                "overfit_gap": metrics.get("overfit_gap", 0.0),
                "per_class_accuracy": metrics.get("per_class_accuracy", []),
                "cross_val_mean": metrics.get("cross_val_mean", 0.0),
                "cross_val_std": metrics.get("cross_val_std", 0.0),
                "budget_remaining": metrics.get("budget_remaining", 0),
                "improved": metrics.get("improved", False),
                "previous_best": metrics.get("previous_best", 0.0),
                "session_summary": session_summary or {},
            },
        )

    @property
    def state(self) -> State:
        return self._state
