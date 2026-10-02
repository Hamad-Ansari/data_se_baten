"""Domain exceptions with user-friendly messages.

Nothing in the UI layer ever shows a raw traceback: every failure is converted
into a :class:`DataSenseError` (or handled explicitly) whose ``user_message`` is
safe to display.  Technical detail is kept in ``technical_detail`` and written
to the logs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class DataSenseError(Exception):
    """Base class for every expected failure in the platform."""

    default_user_message = "Something went wrong while processing your request."
    status_code = 400

    def __init__(
        self,
        message: str = "",
        *,
        user_message: Optional[str] = None,
        technical_detail: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message or user_message or self.default_user_message)
        self.message = message or user_message or self.default_user_message
        self.user_message = user_message or self.default_user_message
        self.technical_detail = technical_detail
        self.context = context or {}
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the error for the REST API in a traceback-free shape."""
        return {
            "error": type(self).__name__,
            "message": self.user_message,
            "detail": self.technical_detail,
            "context": self.context,
        }


class ConfigurationError(DataSenseError):
    default_user_message = "The platform is misconfigured. Please check the settings."


class UnsupportedFormatError(DataSenseError):
    default_user_message = (
        "This file format is not supported. Please upload CSV, TSV, XLSX, JSON, "
        "Parquet, TXT or a ZIP archive containing one of those."
    )


class DatasetTooLargeError(DataSenseError):
    default_user_message = "The dataset is larger than the configured limit."
    status_code = 413


class EmptyDatasetError(DataSenseError):
    default_user_message = "The dataset does not contain any rows."


class DatasetNotFoundError(DataSenseError):
    default_user_message = "The dataset could not be found."
    status_code = 404


class RunNotFoundError(DataSenseError):
    default_user_message = "The requested analysis run could not be found."
    status_code = 404


class SchemaError(DataSenseError):
    default_user_message = "The dataset structure could not be interpreted."


class TargetNotFoundError(DataSenseError):
    default_user_message = "The requested target column is not present in the dataset."


class TooFewRowsError(DataSenseError):
    default_user_message = "There is not enough data to train a reliable model."


class ConstantTargetError(DataSenseError):
    default_user_message = (
        "The target column has a single value, so no model can learn from it."
    )


class InsufficientDataError(DataSenseError):
    default_user_message = "There is not enough data to complete this step."


class PreprocessingError(DataSenseError):
    default_user_message = "The data could not be prepared for modelling."


class TrainingError(DataSenseError):
    default_user_message = "Model training failed. Please review the dataset and try again."


class OptimizationError(DataSenseError):
    default_user_message = "Hyper-parameter optimisation could not be completed."


class EvaluationError(DataSenseError):
    default_user_message = "The model could not be evaluated."


class ExplainabilityError(DataSenseError):
    default_user_message = "The explanation could not be generated for this model."


class QualityGateError(DataSenseError):
    default_user_message = "The model did not pass the quality gate."


class AgentError(DataSenseError):
    default_user_message = "The AI agent could not complete the workflow."


class ToolExecutionError(AgentError):
    default_user_message = "A tool used by the agent failed."


class WorkflowInterrupted(DataSenseError):
    """Raised when a human approval checkpoint pauses the workflow."""

    default_user_message = "The workflow is waiting for your approval before continuing."


class LLMUnavailableError(DataSenseError):
    default_user_message = (
        "The local LLM (Ollama) is not reachable. Deterministic analysis still works; "
        "start Ollama or set OLLAMA_BASE_URL to enable natural-language reasoning."
    )


class InvalidLLMResponseError(DataSenseError):
    default_user_message = "The language model returned an unusable answer."


class ModelNotFoundError(DataSenseError):
    default_user_message = "No trained model is available for this run."
    status_code = 404


class SQLAccessError(DataSenseError):
    default_user_message = "The SQL query was rejected. Only read-only SELECT queries are allowed."


class ReportGenerationError(DataSenseError):
    default_user_message = "The report could not be generated."


def describe_exception(exc: BaseException) -> Dict[str, Any]:
    """Return a safe, serialisable description of any exception."""
    if isinstance(exc, DataSenseError):
        return exc.to_dict()
    return {
        "error": type(exc).__name__,
        "message": "An unexpected internal error occurred. The technical details were logged.",
        "detail": str(exc)[:2000],
        "context": {},
    }


def user_message_for(exc: BaseException) -> str:
    """Return the message a normal user should see for ``exc``."""
    if isinstance(exc, DataSenseError):
        return exc.user_message
    return (
        "An unexpected internal error occurred. Please try again - the technical "
        "details have been written to the log files."
    )


__all__ = [
    "AgentError",
    "ConfigurationError",
    "ConstantTargetError",
    "DataSenseError",
    "DatasetNotFoundError",
    "DatasetTooLargeError",
    "EmptyDatasetError",
    "EvaluationError",
    "ExplainabilityError",
    "InsufficientDataError",
    "InvalidLLMResponseError",
    "LLMUnavailableError",
    "ModelNotFoundError",
    "OptimizationError",
    "PreprocessingError",
    "QualityGateError",
    "ReportGenerationError",
    "RunNotFoundError",
    "SQLAccessError",
    "SchemaError",
    "TargetNotFoundError",
    "TooFewRowsError",
    "ToolExecutionError",
    "TrainingError",
    "UnsupportedFormatError",
    "WorkflowInterrupted",
    "describe_exception",
    "user_message_for",
]
