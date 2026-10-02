"""Unsupervised learning: clustering and dimensionality reduction.

Clustering
    K-Means (with an elbow/silhouette sweep), DBSCAN, HDBSCAN, Agglomerative
    clustering and Gaussian mixtures - evaluated with internal indices
    (silhouette, Davies-Bouldin, Calinski-Harabasz) and profiled per cluster.

Dimensionality reduction
    PCA / Truncated SVD / UMAP projections for visual inspection, with the
    explained-variance share reported so the loss of information is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml import evaluation as eval_mod
from ml.column_analysis import is_numeric_series
from ml.feature_engineering import FeaturePlan, build_preprocessor
from ml.registry import get_algorithm
from ml.tasks import TaskType
from utils.serialization import safe_float, to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)

MAX_PROJECTION_POINTS = 3000


@dataclass
class UnsupervisedResult:
    """Result of a clustering / dimensionality-reduction run."""

    task: str
    models: List[Dict[str, Any]]
    best_model: Optional[str]
    best_metrics: Dict[str, Any]
    labels: Optional[List[int]]
    projection: Optional[Dict[str, Any]]
    cluster_profiles: List[Dict[str, Any]]
    feature_names: List[str]
    n_samples: int
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


def _transform_features(df: pd.DataFrame, plan: FeaturePlan) -> Tuple[np.ndarray, List[str], Any]:
    """Fit (on the full dataset - unsupervised tasks have no leakage risk here)."""
    preprocessor = build_preprocessor(df, plan, needs_scaling=True)
    matrix = preprocessor.fit_transform(df)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # pragma: no cover
        names = [f"feature_{index}" for index in range(matrix.shape[1])]
    return matrix, names, preprocessor


def _project(matrix: np.ndarray, method: str = "pca") -> Optional[Dict[str, Any]]:
    """2D projection for visualisation."""
    try:
        if matrix.shape[1] < 2 or len(matrix) < 3:
            return None
        if method == "umap" and _umap_available():
            from umap import UMAP

            projection = UMAP(n_components=2, random_state=get_settings().random_state).fit_transform(matrix)
            explained = None
        else:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=get_settings().random_state)
            projection = pca.fit_transform(matrix)
            explained = [round(float(value), 6) for value in pca.explained_variance_ratio_]
        sample = projection
        indices = np.arange(len(projection))
        if len(projection) > MAX_PROJECTION_POINTS:
            rng = np.random.default_rng(get_settings().random_state)
            indices = np.sort(rng.choice(len(projection), MAX_PROJECTION_POINTS, replace=False))
            sample = projection[indices]
        return {
            "method": "UMAP" if method == "umap" else "PCA",
            "x": [round(float(value), 5) for value in sample[:, 0]],
            "y": [round(float(value), 5) for value in sample[:, 1]],
            "indices": [int(index) for index in indices],
            "explained_variance": explained,
        }
    except Exception as exc:  # pragma: no cover
        logger.debug("Projection failed: %s", exc)
        return None


def _umap_available() -> bool:
    from utils.optional_deps import is_available

    return is_available("umap")


def run_clustering(
    df: pd.DataFrame,
    plan: FeaturePlan,
    *,
    algorithms: Optional[Sequence[str]] = None,
    cluster_range: Sequence[int] = (2, 3, 4, 5, 6, 8),
    projection_method: str = "pca",
) -> UnsupervisedResult:
    """Fit several clustering algorithms and rank them by internal indices."""
    settings = get_settings()
    keys = list(algorithms or ["kmeans", "agglomerative", "gaussian_mixture"])
    if "hdbscan" not in keys and _package_available("hdbscan") and len(df) >= 100:
        keys.append("hdbscan")
    if "dbscan" not in keys and len(df) <= 20_000:
        keys.append("dbscan")

    notes: List[str] = []
    warnings: List[str] = []
    with Stopwatch() as watch:
        matrix, feature_names, _ = _transform_features(df, plan)
        results: List[Dict[str, Any]] = []
        best_labels: Optional[np.ndarray] = None
        best_key: Optional[str] = None
        best_silhouette = -2.0
        per_model_labels: Dict[str, np.ndarray] = {}

        for key in keys:
            try:
                spec = get_algorithm(key)
            except KeyError:
                warnings.append(f"Unknown clustering algorithm '{key}' was skipped.")
                continue
            if not spec.available:
                warnings.append(f"{spec.name} is unavailable ({spec.install_hint}).")
                continue
            try:
                if key == "kmeans":
                    results.extend(_kmeans_sweep(matrix, cluster_range))
                    best_of_family = max(
                        [item for item in results if item["algorithm"] == key and item["metrics"].get("silhouette") is not None],
                        key=lambda item: item["metrics"].get("silhouette") or -2,
                        default=None,
                    )
                    if best_of_family is not None:
                        estimator = get_algorithm("kmeans").builder(
                            {"n_clusters": best_of_family["params"]["n_clusters"]},
                            random_state=settings.random_state,
                        )
                        labels = estimator.fit_predict(matrix)
                        per_model_labels[key] = labels
                        if (best_of_family["metrics"].get("silhouette") or -2) > best_silhouette:
                            best_silhouette = float(best_of_family["metrics"]["silhouette"])
                            best_labels, best_key = labels, key
                    continue
                estimator = spec.builder({}, random_state=settings.random_state)
                labels = estimator.fit_predict(matrix)
                per_model_labels[key] = labels
                metric_scores: Dict[str, float] = {}
                if key == "gaussian_mixture":
                    try:
                        metric_scores["bic"] = float(estimator.bic(matrix))
                    except Exception:  # pragma: no cover
                        pass
                metrics = eval_mod.clustering_metrics(matrix, labels, metric_scores)
                results.append(
                    {
                        "algorithm": key,
                        "name": spec.name,
                        "params": {},
                        "metrics": metrics,
                        "notes": [],
                    }
                )
                silhouette = metrics.get("silhouette")
                if silhouette is not None and float(silhouette) > best_silhouette and int(metrics.get("n_clusters", 0)) >= 2:
                    best_silhouette = float(silhouette)
                    best_labels, best_key = labels, key
            except Exception as exc:
                logger.warning("Clustering algorithm %s failed: %s", key, exc)
                results.append(
                    {
                        "algorithm": key,
                        "name": spec.name,
                        "params": {},
                        "metrics": {},
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    }
                )

        profiles: List[Dict[str, Any]] = []
        projection = None
        if best_labels is not None:
            labels_series = pd.Series(best_labels, name="cluster")
            profiles = cluster_profiles(df, labels_series, feature_names=feature_names)
            projection = _project(matrix, projection_method)
            if projection is not None:
                projection["labels"] = [int(value) for value in best_labels[projection["indices"]]]
            notes.append(
                f"The best segmentation uses {int(pd.Series(best_labels).nunique())} cluster(s) from "
                f"{get_algorithm(best_key).name} (silhouette {best_silhouette:.3f})."
            )
            if best_silhouette < 0.25:
                warnings.append(
                    "The best silhouette score is below 0.25, which usually means the data does not contain "
                    "well-separated groups - treat the clusters as descriptive, not definitive."
                )

        best_metrics = {}
        for item in results:
            if item["algorithm"] == best_key and item["metrics"]:
                if item["algorithm"] == "kmeans":
                    # pick the metrics belonging to the chosen k
                    if best_labels is not None and item["metrics"].get("n_clusters") == int(pd.Series(best_labels).nunique()):
                        best_metrics = item["metrics"]
                else:
                    best_metrics = item["metrics"]
        if not best_metrics and best_labels is not None:
            best_metrics = eval_mod.clustering_metrics(matrix, best_labels)

    return UnsupervisedResult(
        task=TaskType.CLUSTERING.value,
        models=results,
        best_model=best_key,
        best_metrics=best_metrics,
        labels=[int(value) for value in best_labels] if best_labels is not None else None,
        projection=projection,
        cluster_profiles=profiles,
        feature_names=feature_names,
        n_samples=int(len(df)),
        notes=notes,
        warnings=warnings,
        seconds=watch.elapsed_ms / 1000.0,
    )


def _package_available(name: str) -> bool:
    from utils.optional_deps import is_available

    return is_available(name)


def _kmeans_sweep(matrix: np.ndarray, cluster_range: Sequence[int]) -> List[Dict[str, Any]]:
    """Try several k values and evaluate each with internal indices."""
    from sklearn.cluster import KMeans

    settings = get_settings()
    results: List[Dict[str, Any]] = []
    upper = int(min(max(cluster_range), max(2, len(matrix) // 5)))
    for k in [value for value in cluster_range if 2 <= value <= upper]:
        try:
            estimator = KMeans(n_clusters=int(k), n_init=10, random_state=settings.random_state)
            labels = estimator.fit_predict(matrix)
            metrics = eval_mod.clustering_metrics(
                matrix, labels, {"inertia": float(estimator.inertia_)}
            )
            results.append(
                {
                    "algorithm": "kmeans",
                    "name": f"K-Means (k={k})",
                    "params": {"n_clusters": int(k)},
                    "metrics": metrics,
                    "notes": [f"Sweep evaluation for k={k}."],
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("K-Means k=%s failed: %s", k, exc)
    if not results:
        estimator = KMeans(n_clusters=2, n_init=10, random_state=settings.random_state)
        labels = estimator.fit_predict(matrix)
        results.append(
            {
                "algorithm": "kmeans",
                "name": "K-Means (k=2)",
                "params": {"n_clusters": 2},
                "metrics": eval_mod.clustering_metrics(matrix, labels, {"inertia": float(estimator.inertia_)}),
                "notes": [],
            }
        )
    return results


def cluster_profiles(
    df: pd.DataFrame, labels: pd.Series, *, feature_names: Optional[Sequence[str]] = None, max_features: int = 8
) -> List[Dict[str, Any]]:
    """Describe each cluster and the features that distinguish it most."""
    working = df.copy()
    working["__cluster"] = np.asarray(labels)
    overall_mean = working.select_dtypes(include=[np.number]).mean()
    overall_std = working.select_dtypes(include=[np.number]).std(ddof=0).replace(0, np.nan)
    profiles: List[Dict[str, Any]] = []
    total = len(working)
    for cluster, group in working.groupby("__cluster", observed=True):
        numeric = group.select_dtypes(include=[np.number])
        numeric = numeric.drop(columns=[column for column in ["__cluster"] if column in numeric.columns])
        means = numeric.mean()
        z_scores = ((means - overall_mean) / overall_std).dropna()
        top = z_scores.reindex(z_scores.abs().sort_values(ascending=False).index).head(max_features)
        entry: Dict[str, Any] = {
            "cluster": int(cluster),
            "size": int(len(group)),
            "share": round(float(len(group) / max(total, 1)), 6),
            "distinguishing_features": [
                {
                    "feature": str(feature),
                    "cluster_mean": round(float(means[feature]), 6),
                    "overall_mean": round(float(overall_mean.get(feature, float("nan"))), 6),
                    "z_difference": round(float(value), 4),
                    "direction": "higher" if value > 0 else "lower",
                }
                for feature, value in top.items()
            ],
        }
        for column in working.columns:
            if column == "__cluster":
                continue
            if not is_numeric_series(working[column]):
                top_values = group[column].astype(str).value_counts(normalize=True).head(3)
                if not top_values.empty:
                    entry.setdefault("top_categories", {})[str(column)] = [
                        {"value": str(index), "share": round(float(value), 4)} for index, value in top_values.items()
                    ]
        profiles.append(entry)
    return to_jsonable(profiles)


def run_dimensionality_reduction(
    df: pd.DataFrame,
    plan: FeaturePlan,
    *,
    methods: Sequence[str] = ("pca", "truncated_svd"),
) -> UnsupervisedResult:
    """Project the feature space and report the retained variance."""
    settings = get_settings()
    models: List[Dict[str, Any]] = []
    notes: List[str] = []
    warnings: List[str] = []
    projection: Optional[Dict[str, Any]] = None
    matrix: Optional[np.ndarray] = None
    feature_names: List[str] = []
    with Stopwatch() as watch:
        try:
            matrix, feature_names, _ = _transform_features(df, plan)
        except Exception as exc:
            warnings.append(f"Feature transformation failed: {exc}")
            matrix = None
        if matrix is not None:
            for key in methods:
                try:
                    if key == "umap" and not _umap_available():
                        warnings.append("UMAP requires the optional 'umap-learn' package.")
                        continue
                    if key == "umap":
                        from umap import UMAP

                        estimator = UMAP(n_components=2, random_state=settings.random_state)
                        embedding = estimator.fit_transform(matrix)
                        models.append(
                            {
                                "algorithm": "umap",
                                "name": "UMAP",
                                "metrics": {"n_components": 2},
                                "explained_variance": None,
                            }
                        )
                    elif key == "pca":
                        from sklearn.decomposition import PCA

                        estimator = PCA(n_components=min(10, matrix.shape[1]), random_state=settings.random_state)
                        embedding = estimator.fit_transform(matrix)
                        models.append(
                            {
                                "algorithm": "pca",
                                "name": "PCA",
                                "metrics": {
                                    "n_components": int(estimator.n_components_),
                                    "explained_variance": round(float(estimator.explained_variance_ratio_.sum()), 6),
                                    "explained_variance": round(float(estimator.explained_variance_ratio_.sum()), 6),
                                },
                                "explained_variance_per_component": [
                                    round(float(value), 6) for value in estimator.explained_variance_ratio_
                                ],
                                "loadings": _pca_loadings(estimator, feature_names),
                            }
                        )
                    else:
                        from sklearn.decomposition import TruncatedSVD

                        estimator = TruncatedSVD(n_components=min(10, max(2, matrix.shape[1] - 1)),
                                                 random_state=settings.random_state)
                        embedding = estimator.fit_transform(matrix)
                        models.append(
                            {
                                "algorithm": "truncated_svd",
                                "name": "Truncated SVD",
                                "metrics": {
                                    "n_components": int(estimator.n_components),
                                    "explained_variance": round(float(estimator.explained_variance_ratio_.sum()), 6),
                                },
                            }
                        )
                    if projection is None:
                        projection = _project(matrix, "pca")
                        if projection is not None:
                            projection["labels"] = None
                except Exception as exc:
                    warnings.append(f"{key} projection failed: {type(exc).__name__}")
            if models:
                best = max(models, key=lambda item: item["metrics"].get("explained_variance") or 0)
                notes.append(
                    f"{best['name']} retains "
                    f"{(best['metrics'].get('explained_variance') or 0):.1%} of the variance with "
                    f"{best['metrics'].get('n_components')} component(s)."
                )
                if (best["metrics"].get("explained_variance") or 0) < 0.7:
                    warnings.append(
                        "Less than 70% of the variance is retained - the projection is a visual aid, not a "
                        "faithful representation of the data."
                    )
    return UnsupervisedResult(
        task=TaskType.DIMENSIONALITY_REDUCTION.value,
        models=models,
        best_model=models[0]["algorithm"] if models else None,
        best_metrics=models[0]["metrics"] if models else {},
        labels=None,
        projection=projection,
        cluster_profiles=[],
        feature_names=feature_names,
        n_samples=int(len(df)),
        notes=notes,
        warnings=warnings,
        seconds=watch.elapsed_ms / 1000.0,
    )


def _pca_loadings(estimator: Any, feature_names: Sequence[str], top: int = 5) -> List[Dict[str, Any]]:
    """Largest absolute loading of each component (interpretation aid)."""
    try:
        components = np.asarray(estimator.components_)
        rows: List[Dict[str, Any]] = []
        for index in range(min(3, components.shape[0])):
            order = np.argsort(np.abs(components[index]))[::-1][:top]
            rows.append(
                {
                    "component": index + 1,
                    "explained_variance": round(float(estimator.explained_variance_ratio_[index]), 6),
                    "top_loadings": [
                        {
                            "feature": feature_names[position] if position < len(feature_names) else f"feature_{position}",
                            "loading": round(float(components[index][position]), 6),
                        }
                        for position in order
                    ],
                }
            )
        return rows
    except Exception:  # pragma: no cover
        return []


__all__ = [
    "UnsupervisedResult",
    "cluster_profiles",
    "run_clustering",
    "run_dimensionality_reduction",
]
