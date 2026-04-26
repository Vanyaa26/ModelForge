"""Data models for the ModelForge Environment.

Action: agent submits Python training code.
Observation: dataset description + execution results.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ModelforgeAction(Action):
    """Agent submits Python code that trains an ML model."""

    code: str = Field(
        ...,
        description="Python code that trains a model and assigns it to clf, model, pipe, pipeline, or estimator.",
    )


class ModelforgeObservation(Observation):
    """What the agent sees — dataset description + execution results."""

    dataset_name: str = Field(default="", description="Name of the dataset")
    task_type: str = Field(default="classification", description="classification or regression")
    num_samples: int = Field(default=0, description="Total number of samples")
    num_features: int = Field(default=0, description="Number of features")
    num_classes: int = Field(default=0, description="Number of target classes")
    feature_names: list[str] = Field(default_factory=list, description="Feature column names")
    target_names: list[str] = Field(default_factory=list, description="Target class names")
    sample_rows: list[list[float]] = Field(default_factory=list, description="2-3 example rows")
    class_distribution: dict[str, int] = Field(default_factory=dict, description="Class counts")
    baseline_accuracy: float = Field(default=0.0, description="Majority-class baseline accuracy")

    accuracy: float = Field(default=0.0, description="Achieved test accuracy")
    execution_success: bool = Field(default=False, description="Whether code ran without error")
    error_message: str = Field(default="", description="Error message if execution failed")
    train_time: float = Field(default=0.0, description="Training time in seconds")
    reward_breakdown: dict[str, float] = Field(default_factory=dict, description="Per-signal rewards")

    episode_number: int = Field(default=0, description="Current episode number")
    step_number: int = Field(default=0, description="Current step in episode")
