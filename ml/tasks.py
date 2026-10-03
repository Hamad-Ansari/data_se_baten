"""Task taxonomy shared by the ML layer, the agent and the API."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Optional

from config.constants import PRIMARY_METRIC, TASK_LABELS, METRIC_INFO


class TaskType(str, Enum):
    """Every problem type DATA_SE_BATEN can detect and solve."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    TEXT_CLASSIFICATION = "text_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    UNKNOWN = "unknown"

    # ---------------------------------------------------------------- helpers
    @classmethod
    def coerce(cls, value: object) -> "TaskType":
        """Accept an enum, its value or a loose alias."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "classification": cls.MULTICLASS_CLASSIFICATION,
            "binary": cls.BINARY_CLASSIFICATION,
            "binary_classification": cls.BINARY_CLASSIFICATION,
            "multiclass": cls.MULTICLASS_CLASSIFICATION,
            "multiclass_classification": cls.MULTICLASS_CLASSIFICATION,
            "multilabel": cls.MULTILABEL_CLASSIFICATION,
            "text": cls.TEXT_CLASSIFICATION,
            "text_classification": cls.TEXT_CLASSIFICATION,
            "regression": cls.REGRESSION,
            "clustering": cls.CLUSTERING,
            "segmentation": cls.CLUSTERING,
            "forecasting": cls.TIME_SERIES_FORECASTING,
            "time_series": cls.TIME_SERIES_FORECASTING,
            "time_series_forecasting": cls.TIME_SERIES_FORECASTING,
            "anomaly": cls.ANOMALY_DETECTION,
            "anomaly_detection": cls.ANOMALY_DETECTION,
            "outlier_detection": cls.ANOMALY_DETECTION,
            "dimensionality_reduction": cls.DIMENSIONALITY_REDUCTION,
            "reduction": cls.DIMENSIONALITY_REDUCTION,
            "unknown": cls.UNKNOWN,
        }
        return aliases.get(text, cls.UNKNOWN)

    @property
    def label(self) -> str:
        return TASK_LABELS.get(self.value, self.value.replace("_", " ").title())

    @property
    def primary_metric(self) -> str:
        return PRIMARY_METRIC.get(self.value, "accuracy")

    @property
    def supervised(self) -> bool:
        return self in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.TEXT_CLASSIFICATION,
            TaskType.REGRESSION,
            TaskType.TIME_SERIES_FORECASTING,
        }

    @property
    def classification(self) -> bool:
        return self in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.TEXT_CLASSIFICATION,
        }

    @property
    def regression(self) -> bool:
        return self in {TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING}

    @property
    def needs_target(self) -> bool:
        return self.supervised

    @property
    def temporal(self) -> bool:
        return self is TaskType.TIME_SERIES_FORECASTING


def is_classification(value: object) -> bool:
    return TaskType.coerce(value).classification


def is_regression(value: object) -> bool:
    return TaskType.coerce(value).regression


def is_supervised(value: object) -> bool:
    return TaskType.coerce(value).supervised


def metric_direction(metric: str) -> str:
    """Return 'maximize' or 'minimize' for a metric name."""
    info = METRIC_INFO.get(metric)
    if info:
        return info["direction"]
    lowered = metric.lower()
    if any(token in lowered for token in ("rmse", "mae", "mse", "loss", "error", "mape", "davies")):
        return "minimize"
    return "maximize"


def metric_label(metric: str) -> str:
    info = METRIC_INFO.get(metric)
    return info["label"] if info else metric.replace("_", " ").title()


def metric_reason(metric: str) -> str:
    info = METRIC_INFO.get(metric)
    return info["why"] if info else "Standard metric for this task type."


def higher_is_better(metric: str) -> bool:
    return metric_direction(metric) == "maximize"


def all_tasks() -> List[str]:
    return [task.value for task in TaskType]


def task_metric_candidates(value: object) -> Iterable[str]:
    """Metrics computed for a given task (used by evaluation + UI)."""
    task = TaskType.coerce(value)
    if task.classification:
        return ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")
    if task.regression:
        return ("mae", "mse", "rmse", "r2", "mape")
    if task is TaskType.CLUSTERING:
        return ("silhouette", "davies_bouldin", "calinski_harabasz")
    if task is TaskType.ANOMALY_DETECTION:
        return ("anomaly_rate", "score_separation")
    if task is TaskType.DIMENSIONALITY_REDUCTION:
        return ("explained_variance",)
    return ()


__all__ = [
    "TaskType",
    "all_tasks",
    "higher_is_better",
    "is_classification",
    "is_regression",
    "is_supervised",
    "metric_direction",
    "metric_label",
    "metric_reason",
    "task_metric_candidates",
]
