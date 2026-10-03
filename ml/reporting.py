"""Report generation.

Assembles everything the workflow produced (profile, quality, cleaning audit,
EDA, problem detection, algorithm selection, feature plan, split strategy,
experiments, optimisation, evaluation, explainability, error analysis, quality
gate, deployment and monitoring) into a professional Markdown report and a
self-contained HTML document with the interactive charts embedded.

The report only ever states measured numbers - it reads them from the run
artifacts, never from the language model.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config.constants import METRIC_INFO, TASK_LABELS
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.evaluation import selected_model as _selected
from ml.tasks import TaskType, metric_direction, metric_label
from utils.errors import ReportGenerationError
from utils.files import utc_now_iso
from utils.serialization import safe_float, to_jsonable
from utils.text import humanise, number, pct

logger = get_logger(__name__)

SECTION_ORDER = [
    "executive_summary",
    "dataset_overview",
    "data_quality",
    "cleaning",
    "exploratory_analysis",
    "problem_detection",
    "algorithm_selection",
    "feature_engineering",
    "validation_strategy",
    "model_results",
    "optimization",
    "evaluation",
    "explainability",
    "error_analysis",
    "quality_gate",
    "deployment",
    "monitoring",
    "limitations",
    "next_steps",
]


@dataclass
class ReportBundle:
    """Generated report in every supported format."""

    markdown: str
    html: str
    summary: Dict[str, Any]
    sections: List[Dict[str, str]] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)
    figures_embedded: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "summary": self.summary,
                "sections": self.sections,
                "generated_at": self.generated_at,
                "figures_embedded": self.figures_embedded,
            }
        )


# ---------------------------------------------------------------------------
# markdown helpers
# ---------------------------------------------------------------------------
def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a markdown table."""
    if not rows:
        return "_No data available._"
    lines = ["| " + " | ".join(str(header) for header in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _metric_table(metrics: Dict[str, Any], primary: Optional[str] = None) -> str:
    rows = []
    for key, value in metrics.items():
        numeric = safe_float(value)
        if numeric is None:
            continue
        rows.append(
            [
                metric_label(key) + (" (primary)" if key == primary else ""),
                f"{numeric:,.4f}",
                metric_direction(key),
                METRIC_INFO.get(key, {}).get("why", ""),
            ]
        )
    return table(["Metric", "Value", "Better", "Why it is used"], rows)


# ---------------------------------------------------------------------------
# data collection
# ---------------------------------------------------------------------------
def collect_artifacts(store: Any) -> Dict[str, Any]:
    """Read every artifact the workflow may have produced (missing ones are skipped)."""
    names = {
        "profile": "profile.json",
        "quality": "quality_report.json",
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
        "error_analysis": "error_analysis.json",
        "quality_gate": "quality_gate.json",
        "deployment": "deployment.json",
        "monitoring": "monitoring_reference.json",
        "unsupervised": "unsupervised.json",
        "forecast": "forecast.json",
        "anomaly": "anomaly.json",
    }
    artifacts: Dict[str, Any] = {"run": store.get_all() if hasattr(store, "get_all") else dict(store.meta)}
    for key, filename in names.items():
        payload = store.load_json(filename)
        if payload not in (None, {}, []):
            artifacts[key] = payload
    return artifacts


# ---------------------------------------------------------------------------
# section builders
# ---------------------------------------------------------------------------
def _executive_summary(artifacts: Dict[str, Any], narrative: Optional[str]) -> str:
    run = artifacts.get("run", {})
    profile = artifacts.get("profile", {}) or {}
    problem = artifacts.get("problem", {}) or {}
    model = _selected(artifacts.get("evaluation"))
    gate = artifacts.get("quality_gate") or {}
    lines = [
        "## 1. Executive summary",
        "",
        f"**Dataset:** `{run.get('dataset_name', 'n/a')}` - {profile.get('rows', 0):,} rows x "
        f"{profile.get('columns', 0):,} columns.",
        f"**Detected task:** {TASK_LABELS.get(problem.get('task', 'unknown'), 'unknown')} "
        f"(confidence {float(problem.get('confidence') or 0):.0%}).",
        f"**Target:** `{problem.get('target') or 'n/a'}`.",
    ]
    if model:
        primary = model.get("primary_metric") or "primary metric"
        value = safe_float(model.get("primary_value"))
        lines.append(
            f"**Selected model:** {model.get('name', 'n/a')} with {metric_label(primary)} "
            f"{value:,.4f} on the held-out test set."
            if value is not None
            else f"**Selected model:** {model.get('name', 'n/a')}."
        )
    if gate:
        lines.append(
            f"**Quality gate:** {'PASSED' if gate.get('passed') else 'FAILED'} "
            f"(score {float(gate.get('score') or 0):.0f}/100)."
        )
    if narrative:
        lines.extend(["", "### AI analyst narrative", "", narrative.strip()])
    return "\n".join(lines)


def _dataset_overview(artifacts: Dict[str, Any]) -> str:
    profile = artifacts.get("profile")
    if not profile:
        return "## 2. Dataset overview\n\n_No profile artifact is available._"
    rows, columns = profile.get("rows", 0), profile.get("columns", 0)
    memory = profile.get("memory_bytes") or 0
    lines = [
        "## 2. Dataset overview",
        "",
        f"- Shape: **{rows:,} rows x {columns:,} columns** ({memory / 1e6:.1f} MB in memory)",
        f"- Numeric: {len(profile.get('numeric_features', []))} | Categorical: "
        f"{len(profile.get('categorical_features', []))} | Datetime: "
        f"{len(profile.get('datetime_features', []))} | Text: {len(profile.get('text_features', []))}",
        f"- Missing cells: {profile.get('missing_cells', 0):,} ({float(profile.get('missing_pct') or 0):.2%}) "
        f"in {len(profile.get('missing_columns', []))} column(s)",
        f"- Duplicate rows: {profile.get('duplicate_rows', 0):,} "
        f"({float(profile.get('duplicate_pct') or 0):.2%})",
        f"- Identifier-like columns: {', '.join(profile.get('id_columns', [])[:8]) or 'none'}",
        "",
        "**Target candidates**",
        "",
        table(
            ["Column", "Score", "Type", "Distinct", "Why"],
            [
                [item.get("column"), f"{float(item.get('score') or 0):.2f}", item.get("kind"),
                 item.get("unique"), "; ".join(item.get("reasons", [])[:2])]
                for item in profile.get("target_candidates", [])[:5]
            ],
        ),
    ]
    return "\n".join(lines)


def _data_quality(artifacts: Dict[str, Any]) -> str:
    quality = artifacts.get("quality")
    if not quality:
        return "## 3. Data quality\n\n_No quality artifact is available._"
    issues = quality.get("issues", [])
    lines = [
        "## 3. Data quality",
        "",
        f"**Score:** {float(quality.get('score') or 0):.0f}/100 (grade {quality.get('grade', 'n/a')}) - "
        f"{quality.get('summary', '')}",
        "",
        table(
            ["Dimension", "Score"],
            [[humanise(key), f"{float(value) * 100:.1f}%"] for key, value in (quality.get("dimensions") or {}).items()],
        ),
        "",
        "**Detected issues**",
        "",
        table(
            ["Severity", "Category", "Finding", "Column", "Recommended action"],
            [
                [issue.get("severity"), humanise(issue.get("category", "")), issue.get("title"),
                 issue.get("column") or ", ".join(issue.get("columns", [])[:3]), issue.get("recommended_action")]
                for issue in sorted(
                    issues,
                    key=lambda item: {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(
                        item.get("severity", "info"), 5
                    ),
                )[:20]
            ],
        ),
    ]
    return "\n".join(lines)


def _cleaning(artifacts: Dict[str, Any]) -> str:
    log = artifacts.get("cleaning_log")
    if not log:
        return "## 4. Data cleaning\n\n_No cleaning artifact is available._"
    summary = log.get("summary", {})
    lines = [
        "## 4. Data cleaning",
        "",
        f"- Rows: {summary.get('rows_before', 0):,} -> {summary.get('rows_after', 0):,} "
        f"({summary.get('rows_removed', 0):,} removed)",
        f"- Columns: {summary.get('columns_before', 0):,} -> {summary.get('columns_after', 0):,}",
        f"- Values imputed/repaired: {summary.get('cells_imputed', 0):,}",
        f"- Actions applied: {summary.get('actions_applied', 0)} of {summary.get('actions_planned', 0)} planned"
        + (f" ({summary.get('actions_pending_approval', 0)} awaiting approval)"
           if summary.get("actions_pending_approval") else ""),
        f"- Excluded from features: {', '.join(summary.get('feature_exclusions', [])[:10]) or 'none'}",
        "",
        "**Audit log** (detect -> explain -> strategy -> apply -> validate)",
        "",
        table(
            ["Status", "Action", "Columns", "Strategy", "Why", "Validation"],
            [
                [
                    entry.get("status"),
                    humanise(entry.get("action_type", "")),
                    ", ".join(entry.get("columns", [])[:3]),
                    entry.get("strategy"),
                    entry.get("reason"),
                    "passed" if (entry.get("validation") or {}).get("passed") else "see details",
                ]
                for entry in log.get("log", [])
            ],
        ),
        "",
        "The raw upload is never modified: cleaned data is stored as a separate artifact and every "
        "transformation above is reversible because the original file is preserved in `raw/`.",
    ]
    return "\n".join(lines)


def _eda(artifacts: Dict[str, Any]) -> str:
    eda = artifacts.get("eda")
    if not eda:
        return "## 5. Exploratory analysis\n\n_No EDA artifact is available._"
    insights = eda.get("insights", [])
    lines = ["## 5. Exploratory analysis", ""]
    if insights:
        lines.append("**Computed findings**")
        lines.append("")
        for insight in insights[:12]:
            lines.append(f"- **{insight.get('title')}** - {insight.get('text')}")
        lines.append("")
    numeric = eda.get("numeric_summary", [])
    if numeric:
        lines.append("**Numeric feature statistics**")
        lines.append("")
        lines.append(
            table(
                ["Feature", "Mean", "Std", "Min", "Median", "Max", "Skew"],
                [
                    [
                        item.get("column"),
                        _fmt(item.get("mean")), _fmt(item.get("std")), _fmt(item.get("min")),
                        _fmt(item.get("median")), _fmt(item.get("max")), _fmt(item.get("skew")),
                    ]
                    for item in numeric[:15]
                ],
            )
        )
        lines.append("")
    categorical = eda.get("categorical_summary", [])
    if categorical:
        lines.append("**Categorical features**")
        lines.append("")
        lines.append(
            table(
                ["Feature", "Distinct", "Missing", "Top values"],
                [
                    [
                        item.get("column"), item.get("unique"), item.get("missing"),
                        "; ".join(
                            f"{value.get('value')} ({float(value.get('share') or 0):.0%})"
                            for value in (item.get("top_values") or [])[:4]
                        ),
                    ]
                    for item in categorical[:12]
                ],
            )
        )
        lines.append("")
    correlation = eda.get("correlation")
    if correlation:
        lines.append(
            f"**Correlation matrix** computed with {correlation.get('method', 'pearson')} over "
            f"{len(correlation.get('columns', []))} numeric column(s) - see the interactive heatmap."
        )
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = safe_float(value)
    return f"{numeric:,.{digits}f}" if numeric is not None else ""


def _problem(artifacts: Dict[str, Any]) -> str:
    problem = artifacts.get("problem")
    if not problem:
        return "## 6. Problem detection\n\n_No problem-detection artifact is available._"
    lines = [
        "## 6. Problem detection",
        "",
        f"**Detected task:** {TASK_LABELS.get(problem.get('task', 'unknown'), 'unknown')} "
        f"(confidence {float(problem.get('confidence') or 0):.0%})",
        f"**Primary metric:** {metric_label(problem.get('primary_metric', ''))}",
        "",
        "**Evidence**",
        "",
    ]
    lines.extend(f"- {reason}" for reason in problem.get("reasons", []))
    alternatives = problem.get("alternative_tasks") or []
    if alternatives:
        lines.extend(["", "**Alternative interpretations**", ""])
        for alternative in alternatives:
            lines.append(
                f"- {TASK_LABELS.get(alternative.get('task', ''), alternative.get('task'))} "
                f"(confidence {float(alternative.get('confidence') or 0):.0%}): "
                + "; ".join(alternative.get("reasons", [])[:2])
            )
    if problem.get("notes"):
        lines.extend(["", "**Notes**", ""])
        lines.extend(f"- {note}" for note in problem["notes"])
    return "\n".join(lines)


def _selection(artifacts: Dict[str, Any]) -> str:
    selection = artifacts.get("selection")
    if not selection:
        return "## 7. Algorithm selection\n\n_No selection artifact is available._"
    lines = ["## 7. Algorithm selection", "", "**Baselines**", ""]
    lines.append(
        table(["Algorithm", "Why"], [[item.get("name"), "; ".join(item.get("reasons", [])[:2])]
                                     for item in selection.get("baseline", [])])
    )
    lines.extend(["", "**Candidate models (priority order)**", ""])
    lines.append(
        table(
            ["Algorithm", "Score", "Speed", "Interpretability", "Why", "Caution"],
            [
                [item.get("name"), f"{float(item.get('score') or 0):.2f}", item.get("speed"),
                 item.get("interpretability"), "; ".join(item.get("reasons", [])[:2]),
                 "; ".join(item.get("cautions", [])[:1])]
                for item in selection.get("candidates", [])
            ],
        )
    )
    excluded = selection.get("excluded", [])
    if excluded:
        lines.extend(["", "**Excluded and why**", ""])
        lines.append(
            table(["Algorithm", "Reason"],
                  [[item.get("name"), "; ".join(item.get("reasons", [])[:2])] for item in excluded[:10]])
        )
    if selection.get("notes"):
        lines.extend(["", "**Constraints taken into account**", ""])
        lines.extend(f"- {note}" for note in selection["notes"])
    return "\n".join(lines)


def _features(artifacts: Dict[str, Any]) -> str:
    plan = artifacts.get("feature_plan")
    if not plan:
        return "## 8. Feature engineering\n\n_No feature-plan artifact is available._"
    lines = [
        "## 8. Feature engineering",
        "",
        f"- Numeric: {len(plan.get('numeric_features', []))} | Categorical: "
        f"{len(plan.get('categorical_features', []))} | Datetime: {len(plan.get('datetime_features', []))} | "
        f"Text: {len(plan.get('text_features', []))} | Boolean: {len(plan.get('boolean_features', []))}",
        f"- Scaling: {plan.get('scaling')}",
        f"- Missing-value strategy: " + "; ".join(f"{key}: {value}" for key, value in (plan.get("missing_strategy") or {}).items()),
        "",
    ]
    encodings = plan.get("encodings") or {}
    if encodings:
        lines.append("**Encodings**")
        lines.append("")
        lines.append(table(["Column", "Encoding"], [[key, value] for key, value in list(encodings.items())[:25]]))
        lines.append("")
    engineered = plan.get("engineered") or []
    if engineered:
        lines.append("**Engineered features**")
        lines.append("")
        lines.extend(f"- **{item.get('name')}**: {item.get('description')}" for item in engineered)
        lines.append("")
    excluded = plan.get("excluded_features") or []
    if excluded:
        lines.append("**Excluded columns**")
        lines.append("")
        lines.append(table(["Column", "Reason"], [[item.get("column"), item.get("reason")] for item in excluded[:20]]))
        lines.append("")
    if plan.get("leakage_notes"):
        lines.append("**Leakage controls**")
        lines.append("")
        lines.extend(f"- {note}" for note in plan["leakage_notes"])
    return "\n".join(lines)


def _split(artifacts: Dict[str, Any]) -> str:
    split = artifacts.get("split")
    if not split:
        return "## 9. Validation strategy\n\n_No split artifact is available._"
    lines = [
        "## 9. Validation strategy",
        "",
        f"**Method:** {split.get('method')} - {split.get('description')}",
        f"**Cross-validation:** {split.get('cv_method')} with {split.get('n_splits')} fold(s)",
        "",
    ]
    lines.extend(f"- {reason}" for reason in split.get("reasons", []))
    if split.get("warnings"):
        lines.extend(["", "**Warnings**", ""])
        lines.extend(f"- {warning}" for warning in split["warnings"])
    return "\n".join(lines)


def _model_results(artifacts: Dict[str, Any]) -> str:
    experiments = artifacts.get("experiments")
    if not experiments:
        return "## 10. Model results\n\n_No experiment artifact is available._"
    records = experiments.get("experiments", [])
    primary = experiments.get("primary_metric", "accuracy")
    # only the top-ranked models are scored on the held-out test set, and those
    # scores live with the evaluation candidates keyed by experiment id
    evaluated = {
        item.get("experiment_id"): (item.get("metrics") or {})
        for item in ((artifacts.get("evaluation") or {}).get("candidates") or [])
    }
    rows = []
    for record in records:
        metrics = record.get("validation_metrics") or {}
        test_metrics = record.get("test_metrics") or evaluated.get(record.get("experiment_id")) or {}
        rows.append(
            [
                record.get("name"),
                record.get("stage"),
                record.get("status"),
                _fmt(metrics.get(primary)),
                _fmt(test_metrics.get(primary)),
                _fmt(record.get("cv_mean")),
                _fmt(record.get("cv_std")),
                f"{float(record.get('train_seconds') or 0):.2f}s",
                record.get("n_features"),
            ]
        )
    lines = [
        "## 10. Model results",
        "",
        f"Primary metric: **{metric_label(primary)}** ({metric_direction(primary)}).",
        "",
        table(
            ["Model", "Stage", "Status", f"Val {primary}", f"Test {primary}", "CV mean", "CV std", "Train time",
             "# features"],
            rows,
        ),
        "",
        "_Selection uses the validation split; only the top-ranked models are scored on the held-out test "
        "set, so blank test cells were never evaluated there._",
    ]
    return "\n".join(lines)


def _optimization(artifacts: Dict[str, Any]) -> str:
    optimization = artifacts.get("optimization")
    if not optimization:
        return "## 11. Hyper-parameter optimisation\n\n_No optimisation artifact is available._"
    results = optimization if isinstance(optimization, list) else [optimization]
    lines = ["## 11. Hyper-parameter optimisation", ""]
    for result in results:
        if not result:
            continue
        lines.append(
            f"**{result.get('algorithm_name')}** - {result.get('method')} search over "
            f"{result.get('n_trials', 0)} trial(s) in {float(result.get('duration_seconds') or 0):.1f}s."
        )
        baseline = safe_float(result.get("baseline_value"))
        best = safe_float(result.get("best_value"))
        if baseline is not None and best is not None:
            improvement = safe_float(result.get("improvement"))
            lines.append(
                f"- {metric_label(result.get('metric', ''))}: {baseline:,.4f} -> {best:,.4f} "
                f"({improvement:+,.4f}"
                + (f", {float(result.get('improvement_pct') or 0):+.2f}%" if improvement is not None else "")
                + ")"
            )
        if result.get("best_params"):
            lines.append(f"- Best parameters: `{result['best_params']}`")
        for note in result.get("notes", [])[:3]:
            lines.append(f"- {note}")
        lines.append("")
    lines.append(
        "The search objective is the primary metric measured on the validation split; the test set stays "
        "untouched until the final evaluation."
    )
    return "\n".join(lines)


def _evaluation(artifacts: Dict[str, Any]) -> str:
    evaluation = artifacts.get("evaluation")
    if not evaluation:
        return "## 12. Evaluation\n\n_No evaluation artifact is available._"
    model = _selected(evaluation)
    # the selected entry merges the test metrics into ``metrics`` and also carries
    # them at the top level; the row counts live on the evaluation payload
    metrics = model.get("metrics") or model.get("test_metrics") or {}
    validation_metrics = model.get("validation_metrics") or {}
    primary = model.get("primary_metric") or evaluation.get("primary_metric") or "accuracy"
    test_rows = evaluation.get("test_rows") or model.get("test_rows") or model.get("n_samples") or 0
    validation_rows = evaluation.get("validation_rows") or model.get("validation_rows") or 0
    train_rows = evaluation.get("train_rows") or model.get("training_rows") or 0
    lines = [
        "## 12. Evaluation",
        "",
        f"**Selected model:** {model.get('name')} (`{model.get('key') or model.get('algorithm_key')}`, "
        f"trained as {model.get('stage') or 'n/a'})",
        f"**Held-out test set:** {test_rows:,} rows | "
        f"**Validation:** {validation_rows:,} rows | **Training:** {train_rows:,} rows",
        "",
        "**Test-set metrics**",
        "",
        _metric_table(metrics, primary),
        "",
        "**Validation-set metrics**",
        "",
        _metric_table(validation_metrics, primary),
    ]
    if model.get("cv_mean") is not None:
        folds = len(model.get("cv_scores") or [])
        method = f" ({model['cv_method']})" if model.get("cv_method") else ""
        lines.extend([
            "",
            f"**Cross-validation:** {metric_label(primary)} {float(model['cv_mean']):.4f} "
            f"± {float(model.get('cv_std') or 0):.4f}"
            + (f" across {folds} fold(s)" if folds else "")
            + f"{method}.",
        ])
    confusion = metrics.get("confusion_matrix")
    if confusion:
        labels = confusion.get("labels", [])
        matrix = confusion.get("matrix", [])
        lines.extend(["", "**Confusion matrix (test set)**", ""])
        lines.append(table(["actual \\ predicted"] + list(labels), [
            [labels[index]] + list(row) for index, row in enumerate(matrix)
        ]))
    per_class = metrics.get("per_class")
    if per_class:
        lines.extend(["", "**Per-class performance (test set)**", ""])
        lines.append(
            table(["Class", "Precision", "Recall", "F1", "Support"],
                  [[item.get("class"), _fmt(item.get("precision")), _fmt(item.get("recall")),
                    _fmt(item.get("f1")), item.get("support")] for item in per_class])
        )
    return "\n".join(lines)


def _explainability(artifacts: Dict[str, Any]) -> str:
    explanation = artifacts.get("explanation")
    if not explanation:
        return "## 13. Explainability\n\n_No explainability artifact is available._"
    lines = [
        "## 13. Explainability",
        "",
        f"**Method:** {explanation.get('method')} over {explanation.get('n_explained', 0):,} row(s) and "
        f"{explanation.get('n_features', 0):,} transformed feature(s).",
        "",
    ]
    if explanation.get("narrative"):
        lines.extend([explanation["narrative"], ""])
    ranked = explanation.get("ranked_features") or []
    if ranked:
        lines.append("**Global feature importance (mean |SHAP|)**")
        lines.append("")
        lines.append(
            table(["Feature", "Mean |SHAP|", "Share"],
                  [[item.get("feature"), _fmt(item.get("mean_abs_shap"), 6), f"{float(item.get('share') or 0):.1%}"]
                   for item in ranked[:15]])
        )
        lines.append("")
    local = explanation.get("local_explanations") or []
    if local:
        lines.append("**Local explanations (selected predictions)**")
        lines.append("")
        for entry in local[:3]:
            lines.append(f"- {entry.get('summary')}")
        lines.append("")
    permutation = explanation.get("permutation_importance") or {}
    if permutation:
        top = sorted(permutation.items(), key=lambda item: item[1], reverse=True)[:10]
        lines.append("**Permutation importance (model-agnostic cross-check)**")
        lines.append("")
        lines.append(table(["Column", "Relative drop in primary metric"],
                           [[key, f"{value:.1%}"] for key, value in top]))
        lines.append("")
    lines.append(
        "> Feature importance describes how the model uses each input **on this dataset**. "
        "It is an association, not proof of causation."
    )
    return "\n".join(lines)


def _error_analysis(artifacts: Dict[str, Any]) -> str:
    errors = artifacts.get("error_analysis")
    if not errors or not errors.get("available"):
        return "## 14. Error analysis\n\n_No error analysis artifact is available._"
    lines = ["## 14. Error analysis", "", errors.get("summary", "")]
    segments = errors.get("worst_segments") or []
    if segments:
        lines.extend(["", "**Highest error segments**", ""])
        lines.append(
            table(["Feature", "Segment", "Error rate", "Rows"],
                  [[item.get("feature"), item.get("segment"), pct(item.get("error_rate")), item.get("count")]
                   for item in segments[:10]])
        )
    patterns = errors.get("residual_patterns") or []
    if patterns:
        lines.extend(["", "**Residual patterns**", ""])
        lines.append(
            table(["Feature", "Segment", "Mean residual", "Rows"],
                  [[item.get("feature"), item.get("segment"), _fmt(item.get("mean_residual")), item.get("count")]
                   for item in patterns[:10]])
        )
    return "\n".join(lines)


def _quality_gate(artifacts: Dict[str, Any]) -> str:
    gate = artifacts.get("quality_gate")
    if not gate:
        return "## 15. Quality gate\n\n_No quality-gate artifact is available._"
    lines = [
        "## 15. Quality gate",
        "",
        f"**Result:** {'PASSED' if gate.get('passed') else 'FAILED'} "
        f"(score {float(gate.get('score') or 0):.0f}/100, attempt {gate.get('attempt', 1)}/"
        f"{gate.get('max_attempts', 1)})",
        "",
        table(
            ["Check", "Result", "Evidence", "Recommendation"],
            [[check.get("name"), "PASS" if check.get("passed") else "FAIL", check.get("message"),
              check.get("recommendation") if not check.get("passed") else ""]
             for check in gate.get("checks", [])],
        ),
    ]
    if gate.get("recommendations"):
        lines.extend(["", "**Recommendations**", ""])
        lines.extend(f"- {item}" for item in gate["recommendations"])
    return "\n".join(lines)


def _deployment(artifacts: Dict[str, Any]) -> str:
    deployment = artifacts.get("deployment")
    if not deployment:
        return "## 16. Deployment\n\n_No deployment artifact is available._"
    lines = [
        "## 16. Deployment",
        "",
        f"- Status: **{deployment.get('status')}**",
        f"- Model artifact: `{deployment.get('model_artifact')}`",
        f"- Registered models for this run: {', '.join(deployment.get('available_models', [])[:8]) or 'n/a'}",
        f"- Prediction endpoint: `POST /api/model/predict`",
        f"- Input contract: {deployment.get('input_contract', 'all feature columns')}",
    ]
    if deployment.get("latency_ms") is not None:
        lines.append(f"- Measured single-row latency: {float(deployment['latency_ms']):.2f} ms")
    for note in deployment.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines)


def _monitoring(artifacts: Dict[str, Any]) -> str:
    monitoring = artifacts.get("monitoring")
    if not monitoring:
        return "## 17. Monitoring\n\n_No monitoring artifact is available._"
    features = list((monitoring.get("features") or {}).keys())
    lines = [
        "## 17. Monitoring",
        "",
        f"- Reference profile captured at {monitoring.get('created_at')} from {monitoring.get('rows', 0):,} rows.",
        f"- {len(features)} feature(s) are tracked for drift: {', '.join(features[:10])}"
        + (" ..." if len(features) > 10 else ""),
        "- Drift is measured with the population stability index (PSI) and a Kolmogorov-Smirnov test; "
        "alerts are raised at PSI >= 0.25 and warnings at PSI >= 0.10.",
        "- Prediction requests are logged (input signature, prediction, latency) so traffic and drift can be "
        "reviewed from the Monitoring page.",
    ]
    return "\n".join(lines)


def _limitations(artifacts: Dict[str, Any]) -> str:
    profile = artifacts.get("profile") or {}
    quality = artifacts.get("quality") or {}
    evaluation = artifacts.get("evaluation") or {}
    model = _selected(evaluation)
    gate = artifacts.get("quality_gate") or {}
    items: List[str] = []
    rows = profile.get("rows", 0)
    if rows and rows < 1000:
        items.append(
            f"The dataset contains only {rows:,} rows: metrics have wide confidence intervals and the model "
            "may not generalise beyond this sample."
        )
    if (quality.get("score") or 100) < 70:
        items.append(
            f"Data quality scored {float(quality.get('score') or 0):.0f}/100; unresolved issues (see section 3) "
            "limit how much trust the results deserve."
        )
    if not gate.get("passed"):
        items.append("The quality gate did not pass - see section 15 for the blocking checks.")
    test_rows = evaluation.get("test_rows") or model.get("test_rows") or model.get("n_samples") or 0
    if test_rows:
        items.append(
            f"The test estimate is based on {test_rows:,} rows; re-evaluating on a different period or "
            "sample will shift the numbers."
        )
    items.append(
        "Correlations and feature importances are **associations** in this dataset, not causal effects."
    )
    items.append(
        "The model was trained on a snapshot of the data. Behaviour under drift is monitored but cannot be "
        "guaranteed without regular retraining."
    )
    return "## 18. Limitations\n\n" + "\n".join(f"- {item}" for item in items)


def _next_steps(artifacts: Dict[str, Any]) -> str:
    gate = artifacts.get("quality_gate") or {}
    quality = artifacts.get("quality") or {}
    monitoring = artifacts.get("monitoring") or {}
    steps: List[str] = []
    if not gate.get("passed"):
        steps.extend(
            [f"Address the gate finding: {recommendation}" for recommendation in (gate.get("recommendations") or [])[:3]]
        )
    quality_steps = [
        f"Resolve quality issue: {issue.get('recommended_action')}"
        for issue in (quality.get("issues") or [])
        if issue.get("severity") in {"critical", "high"} and issue.get("recommended_action")
    ]
    steps.extend(quality_steps[:3])
    steps.append("Collect labels for recent production data and re-run the workflow to refresh the model.")
    steps.append("Review the drift dashboard weekly and retrain when PSI exceeds the alert threshold.")
    if monitoring.get("features"):
        steps.append("Confirm the monitored features are still produced by the upstream pipeline.")
    return "## 19. Next steps\n\n" + "\n".join(f"- {step}" for step in steps[:8])


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def build_report(
    store: Any,
    *,
    figures: Optional[Dict[str, Any]] = None,
    narrative: Optional[str] = None,
    include_figures: bool = True,
    title: Optional[str] = None,
) -> ReportBundle:
    """Build the Markdown + HTML report from the run artifacts."""
    try:
        artifacts = collect_artifacts(store)
        run = artifacts.get("run", {})
        settings = get_settings()
        report_title = title or f"{settings.app_name} report - {run.get('dataset_name', 'dataset')}"
        sections: List[Dict[str, str]] = []
        builders = [
            ("executive_summary", "Executive summary", lambda: _executive_summary(artifacts, narrative)),
            ("dataset_overview", "Dataset overview", lambda: _dataset_overview(artifacts)),
            ("data_quality", "Data quality", lambda: _data_quality(artifacts)),
            ("cleaning", "Data cleaning", lambda: _cleaning(artifacts)),
            ("exploratory_analysis", "Exploratory analysis", lambda: _eda(artifacts)),
            ("problem_detection", "Problem detection", lambda: _problem(artifacts)),
            ("algorithm_selection", "Algorithm selection", lambda: _selection(artifacts)),
            ("feature_engineering", "Feature engineering", lambda: _features(artifacts)),
            ("validation_strategy", "Validation strategy", lambda: _split(artifacts)),
            ("model_results", "Model results", lambda: _model_results(artifacts)),
            ("optimization", "Hyper-parameter optimisation", lambda: _optimization(artifacts)),
            ("evaluation", "Evaluation", lambda: _evaluation(artifacts)),
            ("explainability", "Explainability", lambda: _explainability(artifacts)),
            ("error_analysis", "Error analysis", lambda: _error_analysis(artifacts)),
            ("quality_gate", "Quality gate", lambda: _quality_gate(artifacts)),
            ("deployment", "Deployment", lambda: _deployment(artifacts)),
            ("monitoring", "Monitoring", lambda: _monitoring(artifacts)),
            ("limitations", "Limitations", lambda: _limitations(artifacts)),
            ("next_steps", "Next steps", lambda: _next_steps(artifacts)),
        ]
        for key, caption, builder in builders:
            try:
                sections.append({"key": key, "title": caption, "markdown": builder()})
            except Exception as exc:  # pragma: no cover - a section must never break the report
                logger.warning("Report section '%s' failed: %s", key, exc)
                sections.append({"key": key, "title": caption, "markdown": f"_{caption} is unavailable ({type(exc).__name__})._"})

        header = [
            f"# {report_title}",
            "",
            f"_Generated by {settings.app_name} v{settings.app_version} on {utc_now_iso()} "
            f"(run `{run.get('run_id', 'n/a')}`)._",
            "",
        ]
        markdown = "\n\n".join(header + [section["markdown"] for section in sections])
        summary = _report_summary(artifacts, store)
        html = render_html(markdown, figures=figures or {}, title=report_title, summary=summary) if include_figures else render_html(markdown, figures={}, title=report_title, summary=summary)
        bundle = ReportBundle(
            markdown=markdown,
            html=html,
            summary=summary,
            sections=sections,
            figures_embedded=len(figures or {}),
        )
        logger.info("Report generated with %d section(s)", len(sections))
        return bundle
    except Exception as exc:
        raise ReportGenerationError(
            f"Report generation failed: {exc}",
            user_message="The report could not be generated from the available artifacts.",
            technical_detail=str(exc),
        ) from exc


def _report_summary(artifacts: Dict[str, Any], store: Any) -> Dict[str, Any]:
    run = artifacts.get("run", {})
    profile = artifacts.get("profile") or {}
    quality = artifacts.get("quality") or {}
    problem = artifacts.get("problem") or {}
    evaluation = artifacts.get("evaluation") or {}
    model = _selected(evaluation)
    gate = artifacts.get("quality_gate") or {}
    return to_jsonable(
        {
            "run_id": run.get("run_id"),
            "dataset": run.get("dataset_name"),
            "generated_at": utc_now_iso(),
            "rows": profile.get("rows"),
            "columns": profile.get("columns"),
            "quality_score": quality.get("score"),
            "task": problem.get("task"),
            "task_label": TASK_LABELS.get(problem.get("task", ""), ""),
            "target": problem.get("target"),
            "selected_model": model.get("name"),
            "primary_metric": model.get("primary_metric"),
            "primary_value": model.get("primary_value"),
            "test_value": safe_float(
                (model.get("metrics") or model.get("test_metrics") or {})
                .get(model.get("primary_metric") or "")
            ),
            "gate_passed": gate.get("passed"),
            "gate_score": gate.get("score"),
            "best_model": (store.get("model") or {}).get("name") if hasattr(store, "get") else None,
        }
    )


# ---------------------------------------------------------------------------
# markdown -> html (dependency-free)
# ---------------------------------------------------------------------------
def _inline(text: str) -> str:
    escaped = html_lib.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', escaped)
    return escaped


def markdown_to_html(markdown: str) -> str:
    """Convert the small markdown subset used by reports into HTML."""
    lines = markdown.split("\n")
    html_lines: List[str] = []
    in_table = False
    in_list = False
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            html_lines.append("<pre><code>" if in_code else "</code></pre>")
            continue
        if in_code:
            html_lines.append(html_lib.escape(line))
            continue
        if not stripped:
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue
        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if set("".join(cells)) <= {"-", ":", " "}:
                continue
            if not in_table:
                html_lines.append('<table class="report-table"><thead><tr>')
                html_lines.append("".join(f"<th>{_inline(cell)}</th>" for cell in cells))
                html_lines.append("</tr></thead><tbody>")
                in_table = True
            else:
                html_lines.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in cells) + "</tr>")
            continue
        if in_table:
            html_lines.append("</tbody></table>")
            in_table = False
        if stripped.startswith("#"):
            level = min(len(stripped) - len(stripped.lstrip("#")), 6)
            html_lines.append(f"<h{level}>{_inline(stripped[level:].strip())}</h{level}>")
            continue
        if stripped.startswith(("- ", "* ")):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{_inline(stripped[2:])}</li>")
            continue
        if stripped.startswith(">"):
            html_lines.append(f"<blockquote>{_inline(stripped[1:].strip())}</blockquote>")
            continue
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        html_lines.append(f"<p>{_inline(stripped)}</p>")
    if in_table:
        html_lines.append("</tbody></table>")
    if in_list:
        html_lines.append("</ul>")
    if in_code:
        html_lines.append("</code></pre>")
    return "\n".join(html_lines)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {{ --brand:#6366f1; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif; line-height:1.6; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 80px; }}
  header.hero {{ background: linear-gradient(135deg,#4f46e5,#0ea5e9); color:#fff; border-radius:18px;
                 padding:28px 30px; margin-bottom:26px; box-shadow:0 12px 30px rgba(79,70,229,.25); }}
  header.hero h1 {{ margin:0 0 6px; font-size:30px; }}
  header.hero p {{ margin:0; opacity:.92; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:22px 0 30px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:14px 16px; }}
  .card .label {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }}
  .card .value {{ font-size:20px; font-weight:600; margin-top:4px; }}
  section {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:22px 24px; margin-bottom:22px; }}
  h2 {{ font-size:20px; margin-top:0; border-bottom:1px solid var(--line); padding-bottom:8px; }}
  h3 {{ font-size:16px; }}
  table.report-table {{ border-collapse:collapse; width:100%; font-size:13px; margin:12px 0; }}
  table.report-table th, table.report-table td {{ border:1px solid var(--line); padding:8px 10px; text-align:left;
       vertical-align:top; }}
  table.report-table th {{ background:#f1f5f9; font-weight:600; }}
  code {{ background:#f1f5f9; padding:1px 5px; border-radius:5px; font-size:12.5px; }}
  pre {{ background:#0f172a; color:#e2e8f0; padding:14px; border-radius:10px; overflow:auto; }}
  blockquote {{ border-left:4px solid var(--brand); margin:12px 0; padding:6px 14px; color:var(--muted);
                background:#f8fafc; border-radius:0 8px 8px 0; }}
  .figure {{ margin:18px 0; }}
  .figure h3 {{ margin-bottom:6px; }}
  footer {{ text-align:center; color:var(--muted); font-size:12px; margin-top:30px; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <h1>{title}</h1>
    <p>{subtitle}</p>
  </header>
  {cards}
  {body}
  {figures}
  <footer>Generated by DATA_SE_BATEN &middot; all metrics computed with scikit-learn / NumPy / SciPy</footer>
</div>
</body>
</html>
"""


def render_html(
    markdown: str,
    *,
    figures: Optional[Dict[str, Any]] = None,
    title: str = "DATA_SE_BATEN report",
    summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the report as a styled, self-contained HTML document."""
    summary = summary or {}
    cards = ""
    card_items = [
        ("Rows", f"{summary.get('rows'):,}" if summary.get("rows") else "n/a"),
        ("Columns", summary.get("columns") or "n/a"),
        ("Quality score", f"{float(summary.get('quality_score') or 0):.0f}/100" if summary.get("quality_score") else "n/a"),
        ("Task", summary.get("task_label") or "n/a"),
        ("Selected model", summary.get("selected_model") or "n/a"),
        (
            "Primary metric",
            f"{metric_label(summary.get('primary_metric') or '')} "
            f"{(safe_float(summary.get('primary_value')) or 0):.4f}"
            if summary.get("primary_value") is not None else "n/a",
        ),
        ("Quality gate", "passed" if summary.get("gate_passed") else "not passed"),
    ]
    cards = '<div class="cards">' + "".join(
        f'<div class="card"><div class="label">{html_lib.escape(str(label))}</div>'
        f'<div class="value">{html_lib.escape(str(value))}</div></div>'
        for label, value in card_items
    ) + "</div>"

    figure_html = ""
    if figures:
        blocks = []
        for name, figure in figures.items():
            try:
                blocks.append(f'<div class="figure"><h3>{html_lib.escape(name.replace("_", " ").title())}</h3>')
                blocks.append(
                    figure.to_html(full_html=False, include_plotlyjs=False, default_height="420px",
                                   config={"displaylogo": False})
                )
                blocks.append("</div>")
            except Exception as exc:  # pragma: no cover
                blocks.append(f"<p><em>Figure '{name}' could not be rendered ({type(exc).__name__}).</em></p>")
        figure_html = '<section><h2>Charts</h2>' + "".join(blocks) + "</section>"

    body = markdown_to_html(markdown)
    return HTML_TEMPLATE.format(
        title=html_lib.escape(title),
        subtitle=html_lib.escape("Talk to your data. Discover. Analyze. Predict."),
        cards=cards,
        body=body,
        figures=figure_html,
    )


def save_report(store: Any, bundle: ReportBundle, name: str = "report") -> Dict[str, str]:
    """Persist the markdown + HTML report into the run's ``reports/`` folder."""
    markdown_path = store.path(f"{name}.md", "reports")
    html_path = store.path(f"{name}.html", "reports")
    markdown_path.write_text(bundle.markdown, encoding="utf-8")
    html_path.write_text(bundle.html, encoding="utf-8")
    store.save_json(f"{name}_summary.json", bundle.summary, subdir="reports")
    return {"markdown": str(markdown_path), "html": str(html_path)}


__all__ = [
    "HTML_TEMPLATE",
    "ReportBundle",
    "build_report",
    "collect_artifacts",
    "markdown_to_html",
    "render_html",
    "save_report",
    "table",
]
