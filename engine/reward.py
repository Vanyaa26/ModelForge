"""Multi-signal reward for AutoLearn.

Rewards the agent for HOW WELL it managed the training process, not just final accuracy.

5 independent signals:
1. accuracy     — final test accuracy (the main goal)
2. improvement  — how much better than baseline
3. efficiency   — fewer turns used = better (smart decisions, not brute force)
4. diagnosis    — did the agent improve after seeing bad metrics (recovered from problems)
5. no_overfit   — small gap between train and test accuracy
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    accuracy: float = 0.0
    improvement: float = 0.0
    efficiency: float = 0.0
    diagnosis: float = 0.0
    no_overfit: float = 0.0

    @property
    def total(self):
        return self.accuracy + self.improvement + self.efficiency + self.diagnosis + self.no_overfit

    def to_dict(self):
        return {
            "accuracy": round(self.accuracy, 3),
            "improvement": round(self.improvement, 3),
            "efficiency": round(self.efficiency, 3),
            "diagnosis": round(self.diagnosis, 3),
            "no_overfit": round(self.no_overfit, 3),
            "total": round(self.total, 3),
        }


def compute_rewards(session_summary: dict, baseline_accuracy: float) -> RewardBreakdown:
    r = RewardBreakdown()
    best_acc = session_summary["best_accuracy"]
    turns_used = session_summary["budget_used"]
    budget = session_summary["budget_total"]
    history = session_summary["history"]
    improvements = session_summary["improvements_made"]
    crashes = session_summary["crashes"]

    # Signal 1: accuracy (0 to 0.3)
    r.accuracy = round(min(best_acc, 1.0) * 0.3, 3)

    # Signal 2: improvement over baseline (0 to 0.3)
    if baseline_accuracy < 1.0:
        scaled = (best_acc - baseline_accuracy) / (1.0 - baseline_accuracy)
        r.improvement = round(max(0.0, min(1.0, scaled)) * 0.3, 3)

    # Signal 3: efficiency — fewer turns = smarter agent (0 to 0.15)
    if turns_used > 0 and budget > 0:
        efficiency_ratio = 1.0 - (turns_used / budget)
        if best_acc > baseline_accuracy:
            r.efficiency = round(efficiency_ratio * 0.15, 3)

    # Signal 4: diagnosis — recovered from crashes or bad results (0 to 0.15)
    recovered = False
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        if prev.get("error") and curr.get("success") and curr.get("accuracy", 0) > 0:
            recovered = True
        if prev.get("accuracy", 0) < baseline_accuracy and curr.get("accuracy", 0) > baseline_accuracy:
            recovered = True
    if recovered:
        r.diagnosis = 0.15
    elif improvements > 0:
        r.diagnosis = 0.05

    # Signal 5: no overfitting (0 to 0.1)
    last_successful = None
    for h in reversed(history):
        if h.get("success"):
            last_successful = h
            break
    if last_successful:
        gap = abs(last_successful.get("overfit_gap", 0))
        if gap < 0.05:
            r.no_overfit = 0.1
        elif gap < 0.15:
            r.no_overfit = 0.05

    return r
