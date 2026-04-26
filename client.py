"""ModelForge Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ModelforgeAction, ModelforgeObservation


class ModelforgeEnv(EnvClient[ModelforgeAction, ModelforgeObservation, State]):
    """
    Client for the ModelForge Environment.

    Example:
        >>> async with ModelforgeEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     print(result.observation.dataset_name)
        ...
        ...     code = "from sklearn.tree import DecisionTreeClassifier\\n..."
        ...     result = await client.step(ModelforgeAction(code=code))
        ...     print(result.observation.accuracy)
    """

    def _step_payload(self, action: ModelforgeAction) -> Dict:
        return {"code": action.code}

    def _parse_result(self, payload: Dict) -> StepResult[ModelforgeObservation]:
        obs_data = payload.get("observation", {})
        observation = ModelforgeObservation(
            dataset_name=obs_data.get("dataset_name", ""),
            task_type=obs_data.get("task_type", "classification"),
            num_samples=obs_data.get("num_samples", 0),
            num_features=obs_data.get("num_features", 0),
            num_classes=obs_data.get("num_classes", 0),
            feature_names=obs_data.get("feature_names", []),
            target_names=obs_data.get("target_names", []),
            sample_rows=obs_data.get("sample_rows", []),
            class_distribution=obs_data.get("class_distribution", {}),
            baseline_accuracy=obs_data.get("baseline_accuracy", 0.0),
            accuracy=obs_data.get("accuracy", 0.0),
            execution_success=obs_data.get("execution_success", False),
            error_message=obs_data.get("error_message", ""),
            train_time=obs_data.get("train_time", 0.0),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            episode_number=obs_data.get("episode_number", 0),
            step_number=obs_data.get("step_number", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
