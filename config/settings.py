"""Central configuration for DATA_SE_BATEN.

Every configurable value lives here and can be overridden through environment
variables, the ``.env`` file or the Streamlit *Settings* page (which persists
overrides to ``config/user_settings.json``).  Nothing is hard-coded elsewhere
in the code base.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
USER_SETTINGS_FILE: Path = PROJECT_ROOT / "config" / "user_settings.json"


class Settings(BaseSettings):
    """Runtime configuration for the whole platform."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---------------------------------------------------------
    app_name: str = "DATA_SE_BATEN"
    app_tagline: str = "Talk to your data. Discover. Analyze. Predict."
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False

    # --- Logging -------------------------------------------------------------
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    log_json: bool = False
    log_to_file: bool = True

    # --- Storage -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    samples_dir: Path = PROJECT_ROOT / "data" / "samples"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    knowledge_dir: Path = PROJECT_ROOT / "data" / "knowledge"

    # --- Upload / security ---------------------------------------------------
    max_file_size_mb: int = 200
    max_upload_rows: int = 2_000_000
    max_upload_columns: int = 2_000
    allowed_extensions: str = (
        ".csv,.tsv,.txt,.xlsx,.xls,.xlsm,.json,.jsonl,.ndjson,.parquet,.pq,.zip"
    )
    allow_sql_sources: bool = True
    sql_allowed_schemes: str = "sqlite,postgresql,postgres,mysql,mariadb"
    sanitize_sql: bool = True
    sql_preview_rows: int = 100_000

    # --- REST API ------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"
    api_request_timeout: int = 900
    cors_origins: str = "*"
    frontend_base_url: str = "http://localhost:8501"

    # --- LLM / Ollama --------------------------------------------------------
    enable_llm: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout: int = 180
    ollama_temperature: float = 0.2
    ollama_num_ctx: int = 8192
    ollama_num_predict: int = 1024
    ollama_health_cache_seconds: int = 20

    # --- Optional integrations ----------------------------------------------
    enable_mlflow: bool = False
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "data_se_baten"
    enable_rag: bool = False
    enable_knowledge_graph: bool = False
    enable_duckdb: bool = True

    # --- Machine learning defaults ------------------------------------------
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_folds: int = 5
    max_cv_folds: int = 10
    min_rows_for_training: int = 30
    max_rows_in_memory: int = 500_000
    importance_max_features: int = 25
    shap_max_samples: int = 300
    shap_background_samples: int = 100

    # --- AutoML / optimisation ----------------------------------------------
    automl_max_candidates: int = 6
    automl_time_budget_seconds: int = 900
    optuna_enabled: bool = True
    optuna_trials: int = 30
    optuna_timeout_seconds: int = 300
    optuna_quick_trials: int = 12
    optimization_method: str = "optuna"
    early_stopping_rounds: int = 20

    # --- Quality gate --------------------------------------------------------
    gate_min_primary_score: Optional[float] = None
    gate_min_improvement_over_baseline: float = 0.0
    gate_max_overfit_gap: float = 0.20
    gate_min_stability_score: float = 0.55
    gate_max_latency_ms: Optional[float] = None
    gate_require_beats_baseline: bool = True
    gate_max_retries: int = 2

    # --- EDA -----------------------------------------------------------------
    eda_max_charts: int = 12
    eda_max_categories: int = 20
    eda_scatter_max_points: int = 5_000
    profile_correlation_max_columns: int = 60

    # --- Agent ---------------------------------------------------------------
    agent_max_retries: int = 2
    agent_recursion_limit: int = 60
    agent_require_human_approval: bool = True
    agent_auto_clean: bool = True
    agent_require_target_confirmation: bool = False
    agent_chat_history_limit: int = 40

    # --- Monitoring ----------------------------------------------------------
    monitoring_psi_warning: float = 0.1
    monitoring_psi_alert: float = 0.25
    monitoring_max_prediction_log: int = 10_000
    monitoring_retrain_min_new_samples: int = 100

    # ------------------------------------------------------------------ utils
    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @property
    def allowed_extension_set(self) -> set[str]:
        return {
            ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        }

    @property
    def sql_scheme_set(self) -> set[str]:
        return {s.strip().lower() for s in self.sql_allowed_schemes.split(",") if s.strip()}

    @property
    def cors_origin_list(self) -> List[str]:
        raw = self.cors_origins.strip()
        if raw in {"", "*"}:
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb) * 1024 * 1024

    def directory_map(self) -> Dict[str, Path]:
        """Return every writable directory the platform needs."""
        return {
            "data": Path(self.data_dir),
            "uploads": Path(self.uploads_dir),
            "processed": Path(self.processed_dir),
            "samples": Path(self.samples_dir),
            "models": Path(self.models_dir),
            "reports": Path(self.reports_dir),
            "logs": Path(self.log_dir),
            "knowledge": Path(self.knowledge_dir),
        }

    def ensure_directories(self) -> None:
        """Create all storage directories (idempotent)."""
        for directory in self.directory_map().values():
            directory.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> Dict[str, Any]:
        """Configuration subset that is safe to expose through the API/UI."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment,
            "max_file_size_mb": self.max_file_size_mb,
            "allowed_extensions": sorted(self.allowed_extension_set),
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "enable_llm": self.enable_llm,
            "enable_mlflow": self.enable_mlflow,
            "enable_rag": self.enable_rag,
            "enable_knowledge_graph": self.enable_knowledge_graph,
            "random_state": self.random_state,
            "cv_folds": self.cv_folds,
            "optuna_trials": self.optuna_trials,
            "optuna_timeout_seconds": self.optuna_timeout_seconds,
            "automl_max_candidates": self.automl_max_candidates,
            "gate_max_retries": self.gate_max_retries,
            "agent_require_human_approval": self.agent_require_human_approval,
        }

    def model_dump_paths(self) -> Dict[str, str]:  # pragma: no cover - helper
        return {key: str(value) for key, value in self.directory_map().items()}


_SETTINGS_CACHE: Optional[Settings] = None


def _load_user_overrides() -> Dict[str, Any]:
    if not USER_SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(USER_SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):  # pragma: no cover - defensive
        return {}


def get_settings(refresh: bool = False) -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None or refresh:
        overrides = _load_user_overrides()
        settings = Settings(**overrides) if overrides else Settings()
        settings.ensure_directories()
        _SETTINGS_CACHE = settings
    return _SETTINGS_CACHE


def reload_settings() -> Settings:
    """Force a re-read of ``.env`` and user overrides."""
    return get_settings(refresh=True)


def apply_overrides(overrides: Dict[str, Any], persist: bool = True) -> Settings:
    """Apply runtime overrides (used by the Streamlit Settings page).

    Only keys that exist on :class:`Settings` are accepted; unknown keys are
    ignored so a stale user-settings file can never break start-up.
    """
    global _SETTINGS_CACHE
    valid = {k: v for k, v in overrides.items() if k in Settings.model_fields}
    merged = _load_user_overrides()
    merged.update(valid)
    if persist:
        USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = USER_SETTINGS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
        tmp.replace(USER_SETTINGS_FILE)
    _SETTINGS_CACHE = None
    return get_settings()


def update_settings(**overrides: Any) -> Settings:
    """Convenience wrapper around :func:`apply_overrides`."""
    return apply_overrides(overrides)


def public_dict(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Everything the Settings page may display (no secrets, absolute paths as strings)."""
    settings = settings or get_settings()
    payload: Dict[str, Any] = {}
    for name, field in Settings.model_fields.items():
        if name.startswith("sql_") or "password" in name or "secret" in name or "token" in name:
            continue
        value = getattr(settings, name)
        payload[name] = str(value) if isinstance(value, Path) else value
    payload["directories"] = {key: str(path) for key, path in settings.directory_map().items()}
    payload["project_root"] = str(PROJECT_ROOT)
    return payload


__all__ = [
    "PROJECT_ROOT",
    "USER_SETTINGS_FILE",
    "Settings",
    "apply_overrides",
    "get_settings",
    "public_dict",
    "reload_settings",
    "update_settings",
]
