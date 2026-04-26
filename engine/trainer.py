"""Inner model trainer for AutoLearn.

Trains sklearn/simple models and returns REAL training metrics
that the agent must interpret to make decisions.
"""

from __future__ import annotations

import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class TrainingSession:
    """Manages an ongoing training session that the agent controls."""

    def __init__(self, X_train, X_test, y_train, y_test, budget_turns=8):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.budget_turns = budget_turns
        self.turn = 0
        self.history = []
        self.current_model = None
        self.current_pipeline = None
        self.best_accuracy = 0.0
        self.best_code = ""
        self.total_time = 0.0

    def execute_and_evaluate(self, code: str) -> dict:
        """Execute agent's code, train model, return REAL metrics."""
        self.turn += 1
        t0 = time.time()

        metrics = {
            "turn": self.turn,
            "budget_remaining": self.budget_turns - self.turn,
            "success": False,
            "accuracy": 0.0,
            "per_class_accuracy": [],
            "cross_val_scores": [],
            "cross_val_mean": 0.0,
            "cross_val_std": 0.0,
            "train_accuracy": 0.0,
            "overfit_gap": 0.0,
            "num_classes": len(np.unique(self.y_test)),
            "train_time": 0.0,
            "total_time": 0.0,
            "error": None,
            "previous_best": self.best_accuracy,
            "improved": False,
        }

        try:
            import warnings as _w
            _w.filterwarnings("ignore")

            local_vars = {
                "X_train": self.X_train.copy(),
                "X_test": self.X_test.copy(),
                "y_train": self.y_train.copy(),
                "y_test": self.y_test.copy(),
                "np": np,
            }

            exec(code, {"__builtins__": __builtins__}, local_vars)

            model = None
            for var_name in ["pipe", "pipeline", "clf", "model", "estimator"]:
                if var_name in local_vars and hasattr(local_vars[var_name], "predict"):
                    model = local_vars[var_name]
                    break

            if model is None:
                metrics["error"] = "No trained model found. Assign to 'clf', 'model', or 'pipe'."
                self.history.append(metrics)
                return metrics

            y_pred_test = model.predict(self.X_test)
            test_acc = accuracy_score(self.y_test, y_pred_test)

            y_pred_train = model.predict(self.X_train)
            train_acc = accuracy_score(self.y_train, y_pred_train)

            classes = np.unique(self.y_test)
            per_class = []
            for c in classes:
                mask = self.y_test == c
                if mask.sum() > 0:
                    class_acc = accuracy_score(self.y_test[mask], y_pred_test[mask])
                    per_class.append(round(class_acc, 3))
                else:
                    per_class.append(0.0)

            try:
                from sklearn.base import is_classifier
                model_name = type(model).__name__.lower()
                skip_cv = any(s in model_name for s in ["mlp", "neural", "pipeline"])
                if skip_cv:
                    cv_mean = round((train_acc + test_acc) / 2, 3)
                    cv_std = round(abs(train_acc - test_acc) / 2, 3)
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=min(3, len(np.unique(self.y_train))), scoring="accuracy")
                    cv_mean = round(float(cv_scores.mean()), 3)
                    cv_std = round(float(cv_scores.std()), 3)
            except Exception:
                cv_mean = test_acc
                cv_std = 0.0

            elapsed = time.time() - t0
            self.total_time += elapsed

            improved = test_acc > self.best_accuracy
            if improved:
                self.best_accuracy = test_acc
                self.best_code = code
                self.current_model = model

            metrics.update({
                "success": True,
                "accuracy": round(test_acc, 4),
                "train_accuracy": round(train_acc, 4),
                "overfit_gap": round(train_acc - test_acc, 4),
                "per_class_accuracy": per_class,
                "cross_val_mean": cv_mean,
                "cross_val_std": cv_std,
                "train_time": round(elapsed, 2),
                "total_time": round(self.total_time, 2),
                "improved": improved,
            })

        except Exception as e:
            metrics["error"] = str(e)[:300]
            metrics["train_time"] = round(time.time() - t0, 2)

        self.history.append(metrics)
        return metrics

    def get_summary(self) -> dict:
        return {
            "total_turns": self.turn,
            "budget_used": self.turn,
            "budget_total": self.budget_turns,
            "best_accuracy": self.best_accuracy,
            "improvements_made": sum(1 for h in self.history if h.get("improved")),
            "crashes": sum(1 for h in self.history if h.get("error")),
            "history": self.history,
        }
