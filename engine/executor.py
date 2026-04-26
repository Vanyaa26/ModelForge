"""Safe code execution for ModelForge.

Receives Python code as a string, writes to a temp file,
executes in a subprocess with timeout, captures accuracy from stdout.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ExecutionResult:
    success: bool
    accuracy: float
    train_time: float
    stdout: str
    stderr: str
    error: Optional[str]


def execute_training_code(
    code: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    timeout: int = 30,
) -> ExecutionResult:
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "X_train.npy"), X_train)
        np.save(os.path.join(tmpdir, "X_test.npy"), X_test)
        np.save(os.path.join(tmpdir, "y_train.npy"), y_train)
        np.save(os.path.join(tmpdir, "y_test.npy"), y_test)

        wrapper = _build_wrapper(code, tmpdir)
        script_path = os.path.join(tmpdir, "run.py")
        with open(script_path, "w") as f:
            f.write(wrapper)

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={**os.environ, "PYTHONPATH": ""},
            )

            train_time = time.time() - start
            accuracy = _extract_accuracy(result.stdout)

            if result.returncode != 0 and accuracy < 0:
                return ExecutionResult(
                    success=False,
                    accuracy=0.0,
                    train_time=train_time,
                    stdout=result.stdout[-500:],
                    stderr=result.stderr[-500:],
                    error=result.stderr[-300:] if result.stderr else "Non-zero exit code",
                )

            return ExecutionResult(
                success=True,
                accuracy=max(0.0, accuracy),
                train_time=train_time,
                stdout=result.stdout[-500:],
                stderr=result.stderr[-500:],
                error=None,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                accuracy=0.0,
                train_time=timeout,
                stdout="",
                stderr="",
                error=f"Execution timed out after {timeout}s",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                accuracy=0.0,
                train_time=time.time() - start,
                stdout="",
                stderr="",
                error=str(e)[:300],
            )


def _build_wrapper(user_code: str, data_dir: str) -> str:
    return f"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

X_train = np.load("{data_dir}/X_train.npy")
X_test = np.load("{data_dir}/X_test.npy")
y_train = np.load("{data_dir}/y_train.npy")
y_test = np.load("{data_dir}/y_test.npy")

{user_code}
"""


def _extract_accuracy(stdout: str) -> float:
    patterns = [
        r"accuracy:\s*([\d.]+)",
        r"Accuracy:\s*([\d.]+)",
        r"ACCURACY:\s*([\d.]+)",
        r"acc:\s*([\d.]+)",
        r"test accuracy:\s*([\d.]+)",
        r"Test Accuracy:\s*([\d.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout)
        if match:
            val = float(match.group(1))
            if val > 1.0:
                val = val / 100.0
            return min(val, 1.0)
    return -1.0
