"""Shared constants and workflow definitions."""

from __future__ import annotations

from typing import Dict, List, Tuple

APP_NAME = "DATA_SE_BATEN"
APP_TAGLINE = "Talk to your data. Discover. Analyze. Predict."

#: Ordered workflow used by the UI progress indicator and the agent graph.
WORKFLOW_STAGES: Tuple[str, ...] = (
    "ingest",
    "profile",
    "quality",
    "clean",
    "eda",
    "detect",
    "select",
    "features",
    "split",
    "train",
    "optimize",
    "evaluate",
    "explain",
    "gate",
    "report",
    "deploy",
    "monitor",
    "feedback",
)

STAGE_LABELS: Dict[str, str] = {
    "ingest": "Ingest",
    "profile": "Profile",
    "quality": "Quality Check",
    "clean": "Clean",
    "eda": "EDA",
    "detect": "Detect Task",
    "select": "Select Algorithms",
    "features": "Feature Engineering",
    "split": "Train/Test Strategy",
    "train": "Train Models",
    "optimize": "Optimize",
    "evaluate": "Evaluate",
    "explain": "Explain",
    "gate": "Quality Gate",
    "report": "Report",
    "deploy": "Deploy",
    "monitor": "Monitor",
    "feedback": "Feedback",
}

STAGE_ORDER: Dict[str, int] = {stage: index for index, stage in enumerate(WORKFLOW_STAGES)}

STATUS_COMPLETED = "completed"
STATUS_RUNNING = "running"
STATUS_PENDING = "pending"
STATUS_FAILED = "failed"
STATUS_WARNING = "warning"
STATUS_SKIPPED = "skipped"

#: Human readable metric metadata (used by the UI and the report writer).
METRIC_INFO: Dict[str, Dict[str, str]] = {
    "accuracy": {
        "label": "Accuracy",
        "direction": "maximize",
        "why": "Share of correct predictions; meaningful when classes are balanced.",
    },
    "balanced_accuracy": {
        "label": "Balanced accuracy",
        "direction": "maximize",
        "why": "Average recall across classes; robust to moderate class imbalance.",
    },
    "precision": {
        "label": "Precision",
        "direction": "maximize",
        "why": "Of the positive predictions, how many were right - important when false positives are costly.",
    },
    "recall": {
        "label": "Recall",
        "direction": "maximize",
        "why": "Of the real positives, how many were found - important when false negatives are costly.",
    },
    "f1": {
        "label": "F1 score",
        "direction": "maximize",
        "why": "Harmonic mean of precision and recall; the default choice for imbalanced binary problems.",
    },
    "f1_macro": {
        "label": "F1 (macro)",
        "direction": "maximize",
        "why": "Unweighted mean F1 over classes; every class counts equally in multiclass problems.",
    },
    "f1_weighted": {
        "label": "F1 (weighted)",
        "direction": "maximize",
        "why": "F1 weighted by class support; reflects overall performance on the observed distribution.",
    },
    "roc_auc": {
        "label": "ROC AUC",
        "direction": "maximize",
        "why": "Ranking quality across all thresholds, insensitive to the decision threshold.",
    },
    "pr_auc": {
        "label": "PR AUC",
        "direction": "maximize",
        "why": "Precision/recall trade-off across thresholds - the better choice for rare positives.",
    },
    "log_loss": {
        "label": "Log loss",
        "direction": "minimize",
        "why": "Penalises confident wrong probabilities; measures calibration quality.",
    },
    "matthews_corrcoef": {
        "label": "Matthews correlation",
        "direction": "maximize",
        "why": "Balanced single-number summary for binary classification, even with strong imbalance.",
    },
    "mae": {
        "label": "MAE",
        "direction": "minimize",
        "why": "Average absolute error in the target's own units; robust to outliers.",
    },
    "mse": {
        "label": "MSE",
        "direction": "minimize",
        "why": "Squared error; large errors are penalised heavily - useful for variance-sensitive problems.",
    },
    "rmse": {
        "label": "RMSE",
        "direction": "minimize",
        "why": "Square root of MSE, in the target's units; the standard regression error metric.",
    },
    "r2": {
        "label": "R-squared",
        "direction": "maximize",
        "why": "Share of target variance explained relative to the mean predictor.",
    },
    "adjusted_r2": {
        "label": "Adjusted R-squared",
        "direction": "maximize",
        "why": "R-squared penalised by the number of features - compares models of different complexity.",
    },
    "mape": {
        "label": "MAPE",
        "direction": "minimize",
        "why": "Mean absolute percentage error; intuitive when the target is strictly positive.",
    },
    "smape": {
        "label": "sMAPE",
        "direction": "minimize",
        "why": "Symmetric percentage error, safe when the target can be close to zero.",
    },
    "silhouette": {
        "label": "Silhouette score",
        "direction": "maximize",
        "why": "Cluster cohesion vs separation; ranges from -1 (wrong cluster) to 1 (dense, well separated).",
    },
    "davies_bouldin": {
        "label": "Davies-Bouldin",
        "direction": "minimize",
        "why": "Average cluster similarity; lower means better separated clusters.",
    },
    "calinski_harabasz": {
        "label": "Calinski-Harabasz",
        "direction": "maximize",
        "why": "Variance ratio between and within clusters; higher is better.",
    },
    "inertia": {
        "label": "Inertia",
        "direction": "minimize",
        "why": "Within-cluster sum of squares (K-Means objective).",
    },
    "anomaly_rate": {
        "label": "Flagged anomaly rate",
        "direction": "none",
        "why": "Share of records flagged as anomalous; compare with the expected contamination level.",
    },
    "score_separation": {
        "label": "Score separation",
        "direction": "maximize",
        "why": "Standardised distance between normal and flagged anomaly scores; larger means a clearer decision boundary.",
    },
    "explained_variance": {
        "label": "Explained variance",
        "direction": "maximize",
        "why": "Share of total variance retained by the projection.",
    },
}

#: Task -> metric used to rank / gate models.
PRIMARY_METRIC: Dict[str, str] = {
    "binary_classification": "roc_auc",
    "multiclass_classification": "f1_macro",
    "multilabel_classification": "f1_weighted",
    "regression": "rmse",
    "clustering": "silhouette",
    "time_series_forecasting": "rmse",
    "anomaly_detection": "score_separation",
    "dimensionality_reduction": "explained_variance",
    "text_classification": "f1_weighted",
    "unknown": "accuracy",
}

TASK_LABELS: Dict[str, str] = {
    "binary_classification": "Binary classification",
    "multiclass_classification": "Multiclass classification",
    "multilabel_classification": "Multilabel classification",
    "text_classification": "Text classification",
    "regression": "Regression",
    "clustering": "Clustering",
    "time_series_forecasting": "Time series forecasting",
    "anomaly_detection": "Anomaly detection",
    "dimensionality_reduction": "Dimensionality reduction",
    "unknown": "Unknown",
}

#: Severity ordering used by the quality report.
SEVERITY_ORDER: Dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

MAX_PREVIEW_ROWS = 100
DEFAULT_ARTIFACT_NAMES: Dict[str, str] = {
    "profile": "profile.json",
    "quality_report": "quality_report.json",
    "cleaning_plan": "cleaning_plan.json",
    "cleaning_log": "cleaning_log.json",
    "eda": "eda.json",
    "problem": "problem.json",
    "selection": "selection.json",
    "feature_plan": "feature_plan.json",
    "split": "split.json",
    "experiments": "experiments.json",
    "optimization": "optimization.json",
    "evaluation": "evaluation.json",
    "explanation": "explanation.json",
    "quality_gate": "quality_gate.json",
    "deployment": "deployment.json",
    "monitoring": "monitoring_reference.json",
    "report_markdown": "report.md",
    "report_html": "report.html",
}

CHART_COLORS: List[str] = [
    "#6366f1",
    "#22d3ee",
    "#f59e0b",
    "#ef4444",
    "#10b981",
    "#a855f7",
    "#f97316",
    "#0ea5e9",
    "#84cc16",
    "#ec4899",
]

__all__ = [
    "APP_NAME",
    "APP_TAGLINE",
    "CHART_COLORS",
    "DEFAULT_ARTIFACT_NAMES",
    "MAX_PREVIEW_ROWS",
    "METRIC_INFO",
    "PRIMARY_METRIC",
    "SEVERITY_ORDER",
    "STAGE_LABELS",
    "STAGE_ORDER",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "STATUS_SKIPPED",
    "STATUS_WARNING",
    "TASK_LABELS",
    "WORKFLOW_STAGES",
]
