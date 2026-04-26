"""Dataset loading and splitting for ModelForge.

7 classification datasets — 4 sklearn classics + 3 synthetic.
Each returns train/test splits + metadata for the LLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_moons, make_circles, make_classification,
)
from sklearn.model_selection import train_test_split


@dataclass
class DatasetInfo:
    name: str
    task_type: str
    num_samples: int
    num_features: int
    num_classes: int
    feature_names: list[str]
    target_names: list[str]
    sample_rows: list[list[float]]
    class_distribution: dict[str, int]
    baseline_accuracy: float
    X_train: Any = field(repr=False)
    X_test: Any = field(repr=False)
    y_train: Any = field(repr=False)
    y_test: Any = field(repr=False)


DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}

SYNTHETIC_DATASETS = {"moons", "circles", "hard_classify"}


def _load_synthetic(name: str, seed: int):
    if name == "moons":
        X, y = make_moons(500, noise=0.3, random_state=seed)
        return X, y, ["c0", "c1"], ["x1", "x2"]
    if name == "circles":
        X, y = make_circles(500, noise=0.2, factor=0.5, random_state=seed)
        return X, y, ["inner", "outer"], ["x1", "x2"]
    if name == "hard_classify":
        X, y = make_classification(
            800, n_features=20, n_informative=10, n_redundant=5,
            n_classes=4, n_clusters_per_class=2, random_state=seed,
        )
        return X, y, ["c0", "c1", "c2", "c3"], [f"f{i}" for i in range(20)]
    raise ValueError(f"Unknown synthetic dataset: {name}")


def _build_info(name, X, y, target_names, feature_names, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {target_names[int(u)]: int(c) for u, c in zip(unique, counts)}
    baseline_accuracy = round(float(max(counts) / len(y)), 3)
    sample_rows = [[round(v, 2) for v in row] for row in X[:3].tolist()]

    return DatasetInfo(
        name=name, task_type="classification",
        num_samples=len(X), num_features=X.shape[1],
        num_classes=len(target_names),
        feature_names=feature_names, target_names=target_names,
        sample_rows=sample_rows, class_distribution=class_dist,
        baseline_accuracy=baseline_accuracy,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
    )


def get_dataset(name: str, seed: int = 42) -> DatasetInfo:
    if name in SYNTHETIC_DATASETS:
        X, y, targets, features = _load_synthetic(name, seed)
        return _build_info(name, X, y, targets, features, seed)

    loader = DATASET_LOADERS[name]
    data = loader()
    target_names = [str(t) for t in data.target_names]
    feature_names = list(data.feature_names) if hasattr(data, "feature_names") else [f"feature_{i}" for i in range(data.data.shape[1])]
    return _build_info(name, data.data, data.target, target_names, feature_names, seed)


def list_datasets() -> list[str]:
    return list(DATASET_LOADERS.keys()) + sorted(SYNTHETIC_DATASETS)
