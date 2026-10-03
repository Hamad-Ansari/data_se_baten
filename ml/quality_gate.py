"""Model quality gate.

A model only continues to reporting/deployment when it passes an explicit,
auditable set of checks: it must beat the baseline, meet the user's threshold,
not overfit, be stable across folds, behave consistently on unseen data, and
respect latency/interpretability requirements.

A failed gate does not stop the workflow - it produces a *retry plan* that the
agent uses to re-enter optimisation/training (bounded by ``GATE_MAX_RETRIES``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.tasks import TaskType, metric_direction, metric_label
from utils.serialization import safe_float, to_jsonable
from utils.files import utc_now_iso

logger = get_logger(__name__)


@dataclass
class GateCheck:
    """One pass/fail check."""

    name: str
    passed: bool
    severity: str                 # info | low | medium | high
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


@dataclass
class GateResult:
    """Aggregated quality-gate decision."""

    passed: bool
    score: float
    checks: List[GateCheck]
    summary: str
    recommendations: List[str]
    retry_recommended: bool
    retry_focus: List[str]
    attempt: int = 1
    max_attempts: int = 3
    generated_at: str = field(default_factory=utc_now_iso)

    def failures(self) -> List[GateCheck]:
        return [check for check in self.checks if not check.passed]

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "passed": self.passed,
                "score": round(self.score, 2),
                "checks": [check.to_dict() for check in self.checks],
                "summary": self.summary,
                "recommendations": self.recommendations,
                "retry_recommended": self.retry_recommended,
                "retry_focus": self.retry_focus,
                "attempt": self.attempt,
                "max_attempts": self.max_attempts,
                "generated_at": self.generated_at,
                "n_failed": len(self.failures()),
            }
        )

    def to_markdown(self) -> str:
        lines = [
            f"**Quality gate: {'PASSED' if self.passed else 'FAILED'}** "
            f"(score {self.score:.0f}/100, attempt {self.attempt}/{self.max_attempts})",
            "",
            "| Check | Result | Evidence |",
            "|---|---|---|",
        ]
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"| {check.name} | {status} | {check.message} |")
        if self.recommendations:
            lines.append("")
            lines.append("**Recommendations**")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


def evaluate_quality_gate(
    *,
    task: object,
    primary_metric: str,
    best_experiment: Optional[Dict[str, Any]],
    baseline_experiment: Optional[Dict[str, Any]] = None,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    cv_result: Optional[Dict[str, Any]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
    test_rows: int = 0,
    explainability_available: bool = True,
    attempt: int = 1,
) -> GateResult:
    """Run every gate check and decide whether the model may be deployed."""
    settings = get_settings()
    requirements = dict(requirements or {})
    task_type = TaskType.coerce(task)
    direction = metric_direction(primary_metric)
    checks: List[GateCheck] = []

    validation_metrics = (best_experiment or {}).get("validation_metrics") or {}
    test_metrics = (best_experiment or {}).get("test_metrics") or {}
    train_metrics = (best_experiment or {}).get("train_metrics") or {}
    validation_score = safe_float(validation_metrics.get(primary_metric))
    test_score = safe_float(test_metrics.get(primary_metric)) if test_metrics else None
    reference = safe_float(
        (baseline_metrics or {}).get(primary_metric)
        or ((baseline_experiment or {}).get("validation_metrics") or {}).get(primary_metric)
    )
    threshold = safe_float(requirements.get("min_primary_score", settings.gate_min_primary_score))

    # 1 -- an actual model must exist
    checks.append(
        GateCheck(
            name="Model trained",
            passed=best_experiment is not None and validation_score is not None,
            severity="high",
            message=(
                f"{best_experiment.get('name')} produced {metric_label(primary_metric)}="
                f"{validation_score:.4f} on validation." if best_experiment and validation_score is not None
                else "No model produced a usable validation score."
            ),
            value=validation_score,
            recommendation="Check that the dataset has enough rows and that the target is usable.",
        )
    )

    # 2 -- must beat the naive baseline
    if reference is not None and validation_score is not None:
        margin = (validation_score - reference) if direction == "maximize" else (reference - validation_score)
        minimum = safe_float(requirements.get("min_improvement_over_baseline",
                                              settings.gate_min_improvement_over_baseline)) or 0.0
        relative = margin / abs(reference) if reference else None
        pass_baseline = margin > minimum or (settings.gate_require_beats_baseline and margin > 0)
        checks.append(
            GateCheck(
                name="Beats the baseline",
                passed=bool(pass_baseline),
                severity="high",
                message=(
                    f"{metric_label(primary_metric)} {validation_score:.4f} vs baseline {reference:.4f} "
                    f"({margin:+.4f}"
                    + (f", {relative:+.1%} relative" if relative is not None else "")
                    + ")."
                ),
                value=round(float(margin), 6),
                threshold=minimum,
                recommendation=(
                    "The model adds no measurable value over a naive guess. Try different algorithms, better "
                    "features or check that the target is not noise."
                ),
            )
        )
    elif validation_score is not None:
        checks.append(
            GateCheck(
                name="Beats the baseline",
                passed=True,
                severity="low",
                message="No baseline score was available; the check was skipped.",
                recommendation="Train the baseline model to make this comparison explicit.",
            )
        )

    # 3 -- user defined threshold
    if threshold is not None and validation_score is not None:
        passed_threshold = validation_score >= threshold if direction == "maximize" else validation_score <= threshold
        checks.append(
            GateCheck(
                name="Meets the required score",
                passed=bool(passed_threshold),
                severity="high",
                message=(
                    f"Required {metric_label(primary_metric)} "
                    f"{'>=' if direction == 'maximize' else '<='} {threshold:.4f}; achieved {validation_score:.4f}."
                ),
                value=validation_score,
                threshold=threshold,
                recommendation="Tune further, engineer more features or relax the requirement explicitly.",
            )
        )

    # 4 -- overfitting
    train_score = safe_float(train_metrics.get(primary_metric))
    if train_score is not None and validation_score is not None:
        gap = (train_score - validation_score) if direction == "maximize" else (
            (validation_score - train_score) / (abs(validation_score) or 1e-9)
        )
        maximum = safe_float(requirements.get("max_overfit_gap", settings.gate_max_overfit_gap)) or 0.2
        checks.append(
            GateCheck(
                name="No severe overfitting",
                passed=bool(gap <= maximum),
                severity="high" if gap > maximum else "info",
                message=(
                    f"Training {metric_label(primary_metric)} {train_score:.4f} vs validation {validation_score:.4f} "
                    f"(gap {gap:+.4f}, tolerated {maximum:.2f})."
                ),
                value=round(float(gap), 6),
                threshold=maximum,
                recommendation=(
                    "Reduce model complexity (shallower trees, stronger regularisation), add data or use "
                    "cross-validation-based model selection."
                ),
            )
        )

    # 5 -- stability across folds
    if cv_result and cv_result.get("cv_mean") is not None:
        stability = safe_float(cv_result.get("stability"))
        cv_mean = safe_float(cv_result.get("cv_mean"))
        minimum_stability = safe_float(requirements.get("min_stability", settings.gate_min_stability_score)) or 0.55
        passed_stability = stability is None or stability >= minimum_stability
        checks.append(
            GateCheck(
                name="Stable across folds",
                passed=bool(passed_stability),
                severity="medium",
                message=(
                    f"{cv_result.get('cv_method')} {metric_label(primary_metric)} mean {cv_mean:.4f} "
                    f"± {safe_float(cv_result.get('cv_std')) or 0:.4f}"
                    + (f" (stability {stability:.2f})" if stability is not None else "")
                    + f" across {cv_result.get('cv_folds')} fold(s)."
                ),
                value=stability,
                threshold=minimum_stability,
                recommendation=(
                    "Performance varies a lot between folds. Collect more data, simplify the model or use "
                    "repeated cross-validation to confirm the estimate."
                ),
            )
        )

    # 6 -- validation vs test consistency (leakage / overfitting signal)
    if test_score is not None and validation_score is not None:
        denominator = abs(validation_score) or 1e-9
        relative_shift = abs(test_score - validation_score) / denominator
        maximum_shift = safe_float(requirements.get("max_validation_test_shift", 0.25)) or 0.25
        checks.append(
            GateCheck(
                name="Consistent on held-out data",
                passed=bool(relative_shift <= maximum_shift),
                severity="high" if relative_shift > maximum_shift else "info",
                message=(
                    f"Validation {metric_label(primary_metric)} {validation_score:.4f} vs test {test_score:.4f} "
                    f"(relative shift {relative_shift:.1%})."
                ),
                value=round(float(relative_shift), 6),
                threshold=maximum_shift,
                recommendation=(
                    "A large drop on the test set suggests leakage, distribution differences or tuning that "
                    "overfitted the validation set. Re-check the split and any engineered features."
                ),
            )
        )

    # 7 -- latency requirement
    latency_requirement = safe_float(requirements.get("max_latency_ms", settings.gate_max_latency_ms))
    if latency_requirement and latency_ms is not None:
        checks.append(
            GateCheck(
                name="Meets the latency requirement",
                passed=bool(latency_ms <= latency_requirement),
                severity="medium",
                message=f"Prediction latency {latency_ms:.2f} ms vs requirement {latency_requirement:.2f} ms.",
                value=round(float(latency_ms), 4),
                threshold=latency_requirement,
                recommendation="Prefer a faster model (linear/tree ensembles) or batch the predictions.",
            )
        )

    # 8 -- evaluation sample size
    minimum_test = int(requirements.get("min_test_rows", 20))
    checks.append(
        GateCheck(
            name="Enough held-out data",
            passed=bool(test_rows >= minimum_test),
            severity="medium",
            message=f"{test_rows:,} row(s) in the test set (minimum {minimum_test:,}).",
            value=float(test_rows),
            threshold=float(minimum_test),
            recommendation=(
                "With so few test rows the metric is noisy; treat it as indicative, use cross-validation and "
                "collect more data before deploying."
            ),
        )
    )

    # 9 -- interpretability / explainability availability
    interpretability_required = str(requirements.get("interpretability", "")).lower() == "high"
    if interpretability_required:
        algorithm = str((best_experiment or {}).get("name", ""))
        interpretable = any(
            token in algorithm.lower() for token in ("logistic", "linear", "ridge", "lasso", "tree", "elastic")
        )
        checks.append(
            GateCheck(
                name="Interpretability requirement",
                passed=bool(interpretable or explainability_available),
                severity="medium",
                message=(
                    f"Model '{algorithm}' "
                    + ("is directly interpretable." if interpretable else
                       ("is explained with SHAP (available)." if explainability_available else
                        "is not interpretable and no explanation is available."))
                ),
                recommendation="Use an interpretable model or generate SHAP explanations before deploying.",
            )
        )

    # scoring
    weights = {"high": 27.0, "medium": 12.0, "low": 5.0, "info": 0.0}
    penalty = sum(weights.get(check.severity, 5.0) for check in checks if not check.passed)
    score = max(0.0, 100.0 - penalty)
    blocking = [check for check in checks if not check.passed and check.severity in {"high", "medium"}]
    passed = not blocking

    recommendations: List[str] = []
    retry_focus: List[str] = []
    for check in blocking:
        if check.recommendation:
            recommendations.append(f"{check.name}: {check.recommendation}")
        name = check.name.lower()
        if "overfit" in name or "training" in name:
            retry_focus.append("regularise")
        elif "baseline" in name or "required score" in name:
            retry_focus.extend(["features", "algorithm", "optimize"])
        elif "stable" in name or "consistent" in name:
            retry_focus.extend(["cross_validation", "algorithm"])
        elif "latency" in name:
            retry_focus.append("faster_model")
        elif "held-out" in name:
            retry_focus.append("check_leakage")
    seen: set[str] = set()
    retry_focus = [item for item in retry_focus if not (item in seen or seen.add(item))]

    summary = (
        f"Quality gate {'passed' if passed else 'failed'} with score {score:.0f}/100 "
        f"({len(blocking)} blocking issue(s) out of {len(checks)} check(s))."
    )
    logger.info("Quality gate: %s", summary)
    return GateResult(
        passed=passed,
        score=score,
        checks=checks,
        summary=summary,
        recommendations=recommendations,
        retry_recommended=not passed and attempt < settings.gate_max_retries,
        retry_focus=retry_focus,
        attempt=attempt,
        max_attempts=max(1, settings.gate_max_retries),
    )


def gate_from_artifact(payload: Dict[str, Any]) -> Optional[GateResult]:
    """Rebuild a :class:`GateResult` from a stored artifact."""
    if not payload:
        return None
    return GateResult(
        passed=bool(payload.get("passed")),
        score=float(payload.get("score", 0.0)),
        checks=[GateCheck(**check) for check in payload.get("checks", [])],
        summary=str(payload.get("summary", "")),
        recommendations=list(payload.get("recommendations", [])),
        retry_recommended=bool(payload.get("retry_recommended")),
        retry_focus=list(payload.get("retry_focus", [])),
        attempt=int(payload.get("attempt", 1)),
        max_attempts=int(payload.get("max_attempts", 3)),
    )


__all__ = ["GateCheck", "GateResult", "evaluate_quality_gate", "gate_from_artifact"]
