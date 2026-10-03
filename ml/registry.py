"""Algorithm registry.

A single source of truth for every estimator DATA_SE_BATEN can train:
which tasks it supports, how to build it, its hyper-parameter search space, its
runtime cost, whether it needs scaled inputs and which optional dependency it
requires.

The algorithm-selection agent only ever chooses from this registry, so it can
never propose an estimator that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from config.logging_setup import get_logger
from ml.tasks import TaskType
from utils.optional_deps import is_available, pip_name

logger = get_logger(__name__)


@dataclass
class AlgorithmSpec:
    """Description and factory for one estimator."""

    key: str
    name: str
    tasks: List[str]
    builder: Callable[..., Any]
    param_space: Dict[str, Any] = field(default_factory=dict)
    supports_probability: bool = False
    needs_scaling: bool = False
    handles_nan: bool = False
    handles_sparse: bool = True
    interpretability: str = "medium"      # high | medium | low
    speed: str = "medium"                 # fast | medium | slow
    min_samples: int = 30
    max_recommended_samples: Optional[int] = None
    optional_dependency: Optional[str] = None
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    is_baseline: bool = False
    notes: str = ""

    @property
    def available(self) -> bool:
        """Whether the required third-party package is installed."""
        if self.optional_dependency is None:
            return True
        return is_available(self.optional_dependency)

    @property
    def install_hint(self) -> str:
        return f"pip install {pip_name(self.optional_dependency)}" if self.optional_dependency else ""

    def supports(self, task: object) -> bool:
        return TaskType.coerce(task).value in self.tasks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "tasks": list(self.tasks),
            "supports_probability": self.supports_probability,
            "needs_scaling": self.needs_scaling,
            "handles_nan": self.handles_nan,
            "interpretability": self.interpretability,
            "speed": self.speed,
            "min_samples": self.min_samples,
            "max_recommended_samples": self.max_recommended_samples,
            "optional_dependency": self.optional_dependency,
            "available": self.available,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "is_baseline": self.is_baseline,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# builders (imported lazily so optional packages stay optional)
# ---------------------------------------------------------------------------
def _sklearn():
    import sklearn
    return sklearn


def build_logistic_regression(params: Dict[str, Any], random_state: int = 42, **_: Any):
    """Logistic regression across sklearn versions.

    scikit-learn 1.8 deprecated ``penalty`` in favour of ``l1_ratio``; older
    versions only understand ``penalty``.  The search space proposes
    ``l1_ratio``, so translate it when running on an older sklearn.
    """
    from sklearn.linear_model import LogisticRegression
    import sklearn

    params = dict(params or {})
    l1_ratio = params.pop("l1_ratio", None)
    penalty = params.pop("penalty", None)
    if l1_ratio is None and penalty is not None:
        l1_ratio = {"l1": 1.0, "l2": 0.0, "elasticnet": 0.5, "none": 0.0}.get(str(penalty).lower(), 0.0)

    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    defaults: Dict[str, Any] = {"max_iter": 2000, "class_weight": None}
    if l1_ratio not in (None, 0.0):
        defaults["solver"] = "saga"
    if (major, minor) >= (1, 8):
        if l1_ratio is not None:
            defaults["l1_ratio"] = float(l1_ratio)
    else:  # pragma: no cover - depends on the installed sklearn
        if l1_ratio is not None:
            defaults["penalty"] = {0.0: "l2", 1.0: "l1"}.get(float(l1_ratio), "elasticnet")
    defaults.update(params)
    # a legacy config may still carry a solver that cannot handle L1/elastic net
    if defaults.get("l1_ratio") and str(defaults.get("solver", "lbfgs")) not in {"saga", "liblinear"}:
        defaults["solver"] = "saga"
    return LogisticRegression(random_state=random_state, **defaults)


def build_linear_regression(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.linear_model import Ridge

    defaults = dict(alpha=1.0)
    defaults.update(params)
    return Ridge(random_state=random_state, **defaults)


def build_lasso(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.linear_model import Lasso

    defaults = dict(alpha=1.0, max_iter=5000)
    defaults.update(params)
    return Lasso(random_state=random_state, **defaults)


def build_elasticnet(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.linear_model import ElasticNet

    defaults = dict(alpha=1.0, l1_ratio=0.5, max_iter=5000)
    defaults.update(params)
    return ElasticNet(random_state=random_state, **defaults)


def build_knn(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    task = _.get("task", "")
    klass = KNeighborsRegressor if TaskType.coerce(task).regression else KNeighborsClassifier
    defaults = dict(n_neighbors=5, weights="distance", n_jobs=-1)
    defaults.update(params)
    return klass(**defaults)


def build_decision_tree(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    task = _.get("task", "")
    klass = DecisionTreeRegressor if TaskType.coerce(task).regression else DecisionTreeClassifier
    defaults = dict(random_state=random_state)
    defaults.update(params)
    return klass(**defaults)


def build_random_forest(params: Dict[str, Any], random_state: int = 42, n_jobs: int = -1, **_: Any):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    task = _.get("task", "")
    klass = RandomForestRegressor if TaskType.coerce(task).regression else RandomForestClassifier
    defaults = dict(n_estimators=300, random_state=random_state, n_jobs=n_jobs)
    defaults.update(params)
    return klass(**defaults)


def build_extra_trees(params: Dict[str, Any], random_state: int = 42, n_jobs: int = -1, **_: Any):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    task = _.get("task", "")
    klass = ExtraTreesRegressor if TaskType.coerce(task).regression else ExtraTreesClassifier
    defaults = dict(n_estimators=300, random_state=random_state, n_jobs=n_jobs)
    defaults.update(params)
    return klass(**defaults)


def build_svm(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.svm import SVC, SVR

    task = TaskType.coerce(_.get("task", ""))
    if task.regression:
        defaults = dict(kernel="rbf", C=1.0)
        defaults.update(params)
        return SVR(**defaults)
    defaults = dict(kernel="rbf", C=1.0, probability=True, random_state=random_state,
                    class_weight=params.pop("class_weight", None))
    defaults.update(params)
    return SVC(**defaults)


def build_mlp(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    task = TaskType.coerce(_.get("task", ""))
    klass = MLPRegressor if task.regression else MLPClassifier
    defaults = dict(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=random_state)
    defaults.update(params)
    return klass(**defaults)


def build_hist_gradient_boosting(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    task = TaskType.coerce(_.get("task", ""))
    klass = HistGradientBoostingRegressor if task.regression else HistGradientBoostingClassifier
    defaults = dict(random_state=random_state, early_stopping=True)
    defaults.update(params)
    return klass(**defaults)


def build_xgboost(params: Dict[str, Any], random_state: int = 42, n_jobs: int = -1, **_: Any):
    from xgboost import XGBClassifier, XGBRegressor

    task = TaskType.coerce(_.get("task", ""))
    common = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    common.update(params)
    if task.regression:
        return XGBRegressor(**common)
    return XGBClassifier(eval_metric="logloss", **common)


def build_lightgbm(params: Dict[str, Any], random_state: int = 42, n_jobs: int = -1, **_: Any):
    from lightgbm import LGBMClassifier, LGBMRegressor

    task = TaskType.coerce(_.get("task", ""))
    common = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )
    common.update(params)
    klass = LGBMRegressor if task.regression else LGBMClassifier
    return klass(**common)


def build_catboost(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from catboost import CatBoostClassifier, CatBoostRegressor

    task = TaskType.coerce(_.get("task", ""))
    common = dict(iterations=400, learning_rate=0.05, depth=6, random_seed=random_state, verbose=False,
                  allow_writing_files=False)
    common.update(params)
    klass = CatBoostRegressor if task.regression else CatBoostClassifier
    return klass(**common)


def build_dummy(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.dummy import DummyClassifier, DummyRegressor

    task = TaskType.coerce(_.get("task", ""))
    if task.regression:
        defaults = dict(strategy="mean")
        defaults.update(params)
        return DummyRegressor(**defaults)
    defaults = dict(strategy="prior", random_state=random_state)
    defaults.update(params)
    return DummyClassifier(**defaults)


def build_kmeans(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.cluster import KMeans

    defaults = dict(n_clusters=5, n_init=10, random_state=random_state)
    defaults.update(params)
    return KMeans(**defaults)


def build_dbscan(params: Dict[str, Any], **_: Any):
    from sklearn.cluster import DBSCAN

    defaults = dict(eps=0.5, min_samples=5, n_jobs=-1)
    defaults.update(params)
    return DBSCAN(**defaults)


def build_hdbscan(params: Dict[str, Any], **_: Any):
    from hdbscan import HDBSCAN  # optional dependency

    defaults = dict(min_cluster_size=15)
    defaults.update(params)
    return HDBSCAN(**defaults)


def build_agglomerative(params: Dict[str, Any], **_: Any):
    from sklearn.cluster import AgglomerativeClustering

    defaults = dict(n_clusters=5, linkage="ward")
    defaults.update(params)
    return AgglomerativeClustering(**defaults)


def build_gaussian_mixture(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.mixture import GaussianMixture

    defaults = dict(n_components=5, random_state=random_state)
    defaults.update(params)
    return GaussianMixture(**defaults)


def build_isolation_forest(params: Dict[str, Any], random_state: int = 42, n_jobs: int = -1, **_: Any):
    from sklearn.ensemble import IsolationForest

    defaults = dict(n_estimators=200, contamination="auto", random_state=random_state, n_jobs=n_jobs)
    defaults.update(params)
    return IsolationForest(**defaults)


def build_local_outlier_factor(params: Dict[str, Any], n_jobs: int = -1, **_: Any):
    from sklearn.neighbors import LocalOutlierFactor

    defaults = dict(n_neighbors=20, contamination="auto", novelty=True, n_jobs=n_jobs)
    defaults.update(params)
    return LocalOutlierFactor(**defaults)


def build_one_class_svm(params: Dict[str, Any], **_: Any):
    from sklearn.svm import OneClassSVM

    defaults = dict(kernel="rbf", nu=0.05, gamma="scale")
    defaults.update(params)
    return OneClassSVM(**defaults)


def build_pca(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.decomposition import PCA

    defaults = dict(n_components=0.95, random_state=random_state)
    defaults.update(params)
    return PCA(**defaults)


def build_truncated_svd(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from sklearn.decomposition import TruncatedSVD

    defaults = dict(n_components=10, random_state=random_state)
    defaults.update(params)
    return TruncatedSVD(**defaults)


def build_umap(params: Dict[str, Any], random_state: int = 42, **_: Any):
    from umap import UMAP  # optional dependency

    defaults = dict(n_components=2, random_state=random_state)
    defaults.update(params)
    return UMAP(**defaults)


def build_arima(params: Dict[str, Any], **_: Any):
    from statsmodels.tsa.arima.model import ARIMA

    defaults = dict(order=(1, 1, 1))
    defaults.update(params)
    return ARIMA(**defaults)


def build_sarima(params: Dict[str, Any], **_: Any):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    defaults = dict(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    defaults.update(params)
    return SARIMAX(**defaults)


def build_exponential_smoothing(params: Dict[str, Any], **_: Any):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    defaults = dict(trend="add", seasonal=None)
    defaults.update(params)
    return ExponentialSmoothing(**defaults)


def build_prophet(params: Dict[str, Any], **_: Any):
    from prophet import Prophet  # optional dependency

    defaults = dict(daily_seasonality=False)
    defaults.update(params)
    return Prophet(**defaults)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_CLF = [TaskType.BINARY_CLASSIFICATION.value, TaskType.MULTICLASS_CLASSIFICATION.value,
        TaskType.TEXT_CLASSIFICATION.value]
_REG = [TaskType.REGRESSION.value]
_TS = [TaskType.TIME_SERIES_FORECASTING.value]
_CLU = [TaskType.CLUSTERING.value]
_ANO = [TaskType.ANOMALY_DETECTION.value]
_DR = [TaskType.DIMENSIONALITY_REDUCTION.value]

ALGORITHMS: Dict[str, AlgorithmSpec] = {
    # ----------------------------------------------------------- baselines
    "dummy": AlgorithmSpec(
        key="dummy",
        name="Dummy (baseline)",
        tasks=_CLF + _REG,
        builder=build_dummy,
        supports_probability=True,
        interpretability="high",
        speed="fast",
        min_samples=10,
        is_baseline=True,
        strengths=["Shows what a naive guess achieves; any useful model must beat it."],
        weaknesses=["No predictive power by construction."],
        notes="Always trained first so improvements are measurable.",
    ),
    # ------------------------------------------------------ classification
    "logistic_regression": AlgorithmSpec(
        key="logistic_regression",
        name="Logistic Regression",
        tasks=_CLF,
        builder=build_logistic_regression,
        param_space={
            "C": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
            # 0 = L2, 1 = L1, in between = elastic net (sklearn >= 1.8 API)
            "l1_ratio": {"type": "categorical", "choices": [0.0, 1.0, 0.5]},
        },
        supports_probability=True,
        needs_scaling=True,
        interpretability="high",
        speed="fast",
        strengths=["Fast, calibrated probabilities", "Coefficients are directly interpretable",
                   "Strong baseline for linearly separable problems"],
        weaknesses=["Cannot capture non-linear interactions without feature engineering"],
        is_baseline=True,
    ),
    "knn": AlgorithmSpec(
        key="knn",
        name="K-Nearest Neighbours",
        tasks=_CLF + _REG,
        builder=build_knn,
        param_space={
            "n_neighbors": {"type": "int", "low": 3, "high": 31},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            "p": {"type": "categorical", "choices": [1, 2]},
        },
        supports_probability=True,
        needs_scaling=True,
        max_recommended_samples=100_000,
        interpretability="medium",
        speed="medium",
        strengths=["No training phase", "Captures local structure well"],
        weaknesses=["Slow at prediction time", "Sensitive to scaling and irrelevant features"],
    ),
    "decision_tree": AlgorithmSpec(
        key="decision_tree",
        name="Decision Tree",
        tasks=_CLF + _REG,
        builder=build_decision_tree,
        param_space={
            "max_depth": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 50},
            "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
        },
        supports_probability=True,
        interpretability="high",
        speed="fast",
        strengths=["Fully interpretable rules", "No scaling required"],
        weaknesses=["Overfits easily", "High variance"],
    ),
    "random_forest": AlgorithmSpec(
        key="random_forest",
        name="Random Forest",
        tasks=_CLF + _REG,
        builder=build_random_forest,
        param_space={
            "n_estimators": {"type": "int", "low": 150, "high": 600},
            "max_depth": {"type": "int", "low": 3, "high": 24},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 30},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
        },
        supports_probability=True,
        interpretability="medium",
        speed="medium",
        strengths=["Robust default", "Handles mixed types", "Built-in feature importance"],
        weaknesses=["Large models", "Extrapolates poorly in regression"],
    ),
    "extra_trees": AlgorithmSpec(
        key="extra_trees",
        name="Extra Trees",
        tasks=_CLF + _REG,
        builder=build_extra_trees,
        param_space={
            "n_estimators": {"type": "int", "low": 150, "high": 600},
            "max_depth": {"type": "int", "low": 3, "high": 24},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 30},
        },
        supports_probability=True,
        interpretability="medium",
        speed="medium",
        strengths=["Faster than Random Forest", "Lower variance"],
        weaknesses=["More bias than Random Forest"],
    ),
    "hist_gradient_boosting": AlgorithmSpec(
        key="hist_gradient_boosting",
        name="Gradient Boosting (sklearn)",
        tasks=_CLF + _REG,
        builder=build_hist_gradient_boosting,
        param_space={
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_iter": {"type": "int", "low": 100, "high": 800},
            "max_leaf_nodes": {"type": "int", "low": 15, "high": 63},
            "l2_regularization": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
        },
        supports_probability=True,
        handles_nan=True,
        interpretability="medium",
        speed="medium",
        strengths=["Strong accuracy with little tuning", "Native NaN support"],
        weaknesses=["Slower than LightGBM on wide data"],
    ),
    "xgboost": AlgorithmSpec(
        key="xgboost",
        name="XGBoost",
        tasks=_CLF + _REG,
        builder=build_xgboost,
        param_space={
            "n_estimators": {"type": "int", "low": 200, "high": 900},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        },
        supports_probability=True,
        handles_nan=True,
        interpretability="medium",
        speed="medium",
        optional_dependency="xgboost",
        strengths=["State of the art on tabular data", "Regularisation and early stopping"],
        weaknesses=["More hyper-parameters to tune", "Less interpretable than linear models"],
    ),
    "lightgbm": AlgorithmSpec(
        key="lightgbm",
        name="LightGBM",
        tasks=_CLF + _REG,
        builder=build_lightgbm,
        param_space={
            "n_estimators": {"type": "int", "low": 200, "high": 900},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 127},
            "min_child_samples": {"type": "int", "low": 5, "high": 60},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        },
        supports_probability=True,
        handles_nan=True,
        interpretability="medium",
        speed="fast",
        optional_dependency="lightgbm",
        strengths=["Very fast on large datasets", "Excellent accuracy"],
        weaknesses=["Can overfit small datasets without regularisation"],
    ),
    "catboost": AlgorithmSpec(
        key="catboost",
        name="CatBoost",
        tasks=_CLF + _REG,
        builder=build_catboost,
        param_space={
            "iterations": {"type": "int", "low": 200, "high": 800},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "depth": {"type": "int", "low": 4, "high": 10},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 20.0},
        },
        supports_probability=True,
        interpretability="medium",
        speed="medium",
        optional_dependency="catboost",
        strengths=["Excellent with high-cardinality categoricals", "Robust defaults"],
        weaknesses=["Larger models and slower training"],
    ),
    "svm": AlgorithmSpec(
        key="svm",
        name="Support Vector Machine",
        tasks=_CLF + _REG,
        builder=build_svm,
        param_space={
            "C": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear"]},
        },
        supports_probability=True,
        needs_scaling=True,
        max_recommended_samples=50_000,
        interpretability="low",
        speed="slow",
        strengths=["Effective in high-dimensional spaces", "Kernel trick for non-linear boundaries"],
        weaknesses=["Scales poorly with rows", "Needs tuning and scaling"],
    ),
    "mlp": AlgorithmSpec(
        key="mlp",
        name="Neural Network (MLP)",
        tasks=_CLF + _REG,
        builder=build_mlp,
        param_space={
            "hidden_layer_sizes": {"type": "categorical",
                                   "choices": [(64, 32), (128, 64), (64, 32, 16), (128,)]},
            "alpha": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "learning_rate_init": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        },
        supports_probability=True,
        needs_scaling=True,
        min_samples=200,
        interpretability="low",
        speed="slow",
        strengths=["Captures complex non-linear patterns", "Works well with many rows"],
        weaknesses=["Needs scaling and tuning", "Hard to explain", "Unstable on small data"],
    ),
    # ---------------------------------------------------------- regression
    "linear_regression": AlgorithmSpec(
        key="linear_regression",
        name="Ridge Regression",
        tasks=_REG,
        builder=build_linear_regression,
        param_space={"alpha": {"type": "float", "low": 1e-4, "high": 1e3, "log": True}},
        needs_scaling=True,
        interpretability="high",
        speed="fast",
        strengths=["Interpretable coefficients", "Very fast", "Strong baseline"],
        weaknesses=["Assumes linearity and no interactions"],
        is_baseline=True,
    ),
    "lasso": AlgorithmSpec(
        key="lasso",
        name="Lasso Regression",
        tasks=_REG,
        builder=build_lasso,
        param_space={"alpha": {"type": "float", "low": 1e-5, "high": 10.0, "log": True}},
        needs_scaling=True,
        interpretability="high",
        speed="fast",
        strengths=["Automatic feature selection (L1)"],
        weaknesses=["Unstable with correlated features"],
    ),
    "elasticnet": AlgorithmSpec(
        key="elasticnet",
        name="ElasticNet Regression",
        tasks=_REG,
        builder=build_elasticnet,
        param_space={
            "alpha": {"type": "float", "low": 1e-5, "high": 10.0, "log": True},
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0},
        },
        needs_scaling=True,
        interpretability="high",
        speed="fast",
        strengths=["Combines L1 and L2", "Handles correlated features better than Lasso"],
        weaknesses=["Two hyper-parameters to tune"],
    ),
    # ---------------------------------------------------------- clustering
    "kmeans": AlgorithmSpec(
        key="kmeans",
        name="K-Means",
        tasks=_CLU,
        builder=build_kmeans,
        param_space={
            "n_clusters": {"type": "int", "low": 2, "high": 12},
            "init": {"type": "categorical", "choices": ["k-means++", "random"]},
        },
        needs_scaling=True,
        interpretability="medium",
        speed="fast",
        strengths=["Fast and scalable", "Easy to interpret centroids"],
        weaknesses=["Assumes spherical clusters of similar size", "Needs k"],
    ),
    "dbscan": AlgorithmSpec(
        key="dbscan",
        name="DBSCAN",
        tasks=_CLU,
        builder=build_dbscan,
        param_space={
            "eps": {"type": "float", "low": 0.1, "high": 5.0, "log": True},
            "min_samples": {"type": "int", "low": 3, "high": 30},
        },
        needs_scaling=True,
        max_recommended_samples=50_000,
        interpretability="medium",
        speed="medium",
        strengths=["Finds arbitrary shapes", "Marks noise explicitly"],
        weaknesses=["Sensitive to eps and density differences"],
    ),
    "hdbscan": AlgorithmSpec(
        key="hdbscan",
        name="HDBSCAN",
        tasks=_CLU,
        builder=build_hdbscan,
        param_space={"min_cluster_size": {"type": "int", "low": 5, "high": 60}},
        needs_scaling=True,
        max_recommended_samples=100_000,
        interpretability="medium",
        speed="medium",
        optional_dependency="hdbscan",
        strengths=["Handles varying densities", "Little parameter tuning"],
        weaknesses=["Requires the optional hdbscan package"],
    ),
    "agglomerative": AlgorithmSpec(
        key="agglomerative",
        name="Agglomerative Clustering",
        tasks=_CLU,
        builder=build_agglomerative,
        param_space={
            "n_clusters": {"type": "int", "low": 2, "high": 12},
            "linkage": {"type": "categorical", "choices": ["ward", "complete", "average"]},
        },
        needs_scaling=True,
        max_recommended_samples=30_000,
        interpretability="medium",
        speed="slow",
        strengths=["Hierarchical structure", "No centroid shape assumption"],
        weaknesses=["O(n²) memory", "Needs a distance threshold or k"],
    ),
    "gaussian_mixture": AlgorithmSpec(
        key="gaussian_mixture",
        name="Gaussian Mixture",
        tasks=_CLU,
        builder=build_gaussian_mixture,
        param_space={
            "n_components": {"type": "int", "low": 2, "high": 12},
            "covariance_type": {"type": "categorical", "choices": ["full", "tied", "diag", "spherical"]},
        },
        needs_scaling=True,
        interpretability="medium",
        speed="medium",
        strengths=["Soft cluster assignments", "Models elliptical clusters"],
        weaknesses=["Assumes Gaussian components"],
    ),
    # --------------------------------------------------- anomaly detection
    "isolation_forest": AlgorithmSpec(
        key="isolation_forest",
        name="Isolation Forest",
        tasks=_ANO,
        builder=build_isolation_forest,
        param_space={
            "n_estimators": {"type": "int", "low": 100, "high": 400},
            "max_samples": {"type": "categorical", "choices": ["auto", 0.5, 0.8]},
            "contamination": {"type": "float", "low": 0.01, "high": 0.2},
        },
        interpretability="medium",
        speed="fast",
        strengths=["Fast, scales well", "No assumptions about the distribution"],
        weaknesses=["Contamination must be specified"],
        is_baseline=True,
    ),
    "local_outlier_factor": AlgorithmSpec(
        key="local_outlier_factor",
        name="Local Outlier Factor",
        tasks=_ANO,
        builder=build_local_outlier_factor,
        param_space={
            "n_neighbors": {"type": "int", "low": 5, "high": 40},
            "contamination": {"type": "float", "low": 0.01, "high": 0.2},
        },
        needs_scaling=True,
        max_recommended_samples=50_000,
        interpretability="medium",
        speed="medium",
        strengths=["Detects local density anomalies"],
        weaknesses=["Slow on large data", "Sensitive to scaling"],
    ),
    "one_class_svm": AlgorithmSpec(
        key="one_class_svm",
        name="One-Class SVM",
        tasks=_ANO,
        builder=build_one_class_svm,
        param_space={
            "nu": {"type": "float", "low": 0.01, "high": 0.3},
            "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        },
        needs_scaling=True,
        max_recommended_samples=30_000,
        interpretability="low",
        speed="slow",
        strengths=["Works well in high dimensions"],
        weaknesses=["Scales poorly with rows"],
    ),
    # ------------------------------------------ dimensionality reduction
    "pca": AlgorithmSpec(
        key="pca",
        name="PCA",
        tasks=_DR + _CLU,
        builder=build_pca,
        param_space={"n_components": {"type": "categorical", "choices": [0.9, 0.95, 0.99, 2, 10]}},
        needs_scaling=True,
        interpretability="medium",
        speed="fast",
        strengths=["Fast, deterministic", "Removes correlated redundancy"],
        weaknesses=["Components are linear combinations (harder to interpret)"],
    ),
    "truncated_svd": AlgorithmSpec(
        key="truncated_svd",
        name="Truncated SVD",
        tasks=_DR + _CLU,
        builder=build_truncated_svd,
        param_space={"n_components": {"type": "int", "low": 2, "high": 50}},
        handles_sparse=True,
        interpretability="medium",
        speed="fast",
        strengths=["Works on sparse matrices (text)"],
        weaknesses=["Less interpretable than PCA"],
    ),
    "umap": AlgorithmSpec(
        key="umap",
        name="UMAP",
        tasks=_DR,
        builder=build_umap,
        param_space={
            "n_neighbors": {"type": "int", "low": 5, "high": 50},
            "min_dist": {"type": "float", "low": 0.0, "high": 0.9},
        },
        optional_dependency="umap",
        needs_scaling=True,
        max_recommended_samples=100_000,
        interpretability="low",
        speed="slow",
        strengths=["Preserves local and global structure for visualisation"],
        weaknesses=["Requires umap-learn (numba)", "Stochastic"],
    ),
    # ------------------------------------------------------- time series
    "naive": AlgorithmSpec(
        key="naive",
        name="Naive forecast (last value)",
        tasks=_TS,
        builder=lambda params, **_: _NaiveForecaster(),
        interpretability="high",
        speed="fast",
        is_baseline=True,
        strengths=["The benchmark every forecasting model must beat"],
        weaknesses=["No trend or seasonality awareness"],
    ),
    "seasonal_naive": AlgorithmSpec(
        key="seasonal_naive",
        name="Seasonal naive",
        tasks=_TS,
        builder=lambda params, **_: _SeasonalNaiveForecaster(season_length=params.get("season_length", 7)),
        param_space={"season_length": {"type": "int", "low": 2, "high": 30}},
        interpretability="high",
        speed="fast",
        strengths=["Strong baseline for seasonal series"],
        weaknesses=["Needs a detected season length"],
    ),
    "arima": AlgorithmSpec(
        key="arima",
        name="ARIMA",
        tasks=_TS,
        builder=build_arima,
        param_space={
            "order": {"type": "categorical", "choices": [(1, 0, 0), (1, 1, 1), (2, 1, 1), (0, 1, 1)]},
        },
        interpretability="high",
        speed="medium",
        optional_dependency="statsmodels",
        max_recommended_samples=20_000,
        strengths=["Classical, interpretable statistical model"],
        weaknesses=["Assumes linear autocorrelation structure"],
    ),
    "sarima": AlgorithmSpec(
        key="sarima",
        name="SARIMA",
        tasks=_TS,
        builder=build_sarima,
        param_space={
            "order": {"type": "categorical", "choices": [(1, 1, 1), (1, 1, 0), (0, 1, 1)]},
            "seasonal_order": {"type": "categorical",
                               "choices": [(1, 1, 1, 7), (1, 1, 1, 12), (0, 1, 1, 7)]},
        },
        interpretability="high",
        speed="slow",
        optional_dependency="statsmodels",
        max_recommended_samples=10_000,
        strengths=["Models seasonality explicitly"],
        weaknesses=["Slow to fit", "Requires enough history"],
    ),
    "exponential_smoothing": AlgorithmSpec(
        key="exponential_smoothing",
        name="Exponential Smoothing",
        tasks=_TS,
        builder=build_exponential_smoothing,
        param_space={
            "trend": {"type": "categorical", "choices": [None, "add"]},
            "seasonal": {"type": "categorical", "choices": [None, "add"]},
        },
        interpretability="high",
        speed="fast",
        optional_dependency="statsmodels",
        strengths=["Fast, weights recent observations more"],
        weaknesses=["Limited to simple trend/seasonality"],
    ),
    "prophet": AlgorithmSpec(
        key="prophet",
        name="Prophet",
        tasks=_TS,
        builder=build_prophet,
        param_space={
            "changepoint_prior_scale": {"type": "float", "low": 0.001, "high": 0.5, "log": True},
            "seasonality_mode": {"type": "categorical", "choices": ["additive", "multiplicative"]},
        },
        interpretability="medium",
        speed="slow",
        optional_dependency="prophet",
        strengths=["Handles holidays and multiple seasonalities"],
        weaknesses=["Heavy dependency", "Needs enough history"],
    ),
    "gb_forecaster": AlgorithmSpec(
        key="gb_forecaster",
        name="Gradient boosting forecaster (lag features)",
        tasks=_TS,
        builder=build_lightgbm,
        param_space={
            "n_estimators": {"type": "int", "low": 200, "high": 700},
            "learning_rate": {"type": "float", "low": 0.02, "high": 0.2, "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 63},
        },
        interpretability="medium",
        speed="medium",
        optional_dependency="lightgbm",
        strengths=["Uses lag/rolling features and exogenous variables", "Often the most accurate"],
        weaknesses=["Needs enough history for lag features"],
    ),
}


class _NaiveForecaster:
    """Last-value forecast (benchmark)."""

    def fit(self, y: Any) -> "_NaiveForecaster":
        import numpy as np

        values = np.asarray(y, dtype=float)
        self.last_value = float(values[-1]) if values.size else 0.0
        return self

    def predict(self, steps: int) -> Any:
        import numpy as np

        return np.repeat(self.last_value, steps)


class _SeasonalNaiveForecaster:
    """Seasonal naive forecast: repeat the last observed season."""

    def __init__(self, season_length: int = 7) -> None:
        self.season_length = max(int(season_length), 2)

    def fit(self, y: Any) -> "_SeasonalNaiveForecaster":
        import numpy as np

        values = np.asarray(y, dtype=float)
        self.season = values[-self.season_length:] if values.size >= self.season_length else values
        self.last_value = float(values[-1]) if values.size else 0.0
        return self

    def predict(self, steps: int) -> Any:
        import numpy as np

        if self.season.size == 0:
            return np.zeros(steps)
        repeats = int(np.ceil(steps / self.season.size))
        return np.tile(self.season, repeats)[:steps]


def get_algorithm(key: str) -> AlgorithmSpec:
    """Return a registered algorithm (raises ``KeyError`` for unknown keys)."""
    if key not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{key}'. Known: {', '.join(sorted(ALGORITHMS))}")
    return ALGORITHMS[key]


def algorithms_for_task(task: object, include_unavailable: bool = False) -> List[AlgorithmSpec]:
    """All algorithms that support ``task`` (installed ones first)."""
    specs = [spec for spec in ALGORITHMS.values() if spec.supports(task)]
    if not include_unavailable:
        specs = [spec for spec in specs if spec.available]
    return specs


def algorithm_names() -> List[str]:
    return sorted(ALGORITHMS)


def registry_table() -> List[Dict[str, Any]]:
    return [spec.to_dict() for spec in ALGORITHMS.values()]


def missing_dependencies(task: Optional[object] = None) -> List[Dict[str, str]]:
    """Trained-capable algorithms whose optional package is not installed."""
    specs = ALGORITHMS.values() if task is None else [s for s in ALGORITHMS.values() if s.supports(task)]
    return [
        {"algorithm": spec.name, "package": spec.optional_dependency or "", "install": spec.install_hint}
        for spec in specs
        if not spec.available and spec.optional_dependency
    ]


__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "algorithm_names",
    "algorithms_for_task",
    "get_algorithm",
    "missing_dependencies",
    "registry_table",
]
