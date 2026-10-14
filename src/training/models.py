"""
Model registry providing factory functions for 10 machine learning models.
Supports both classification and regression tasks, with automatic scaling
pipelines for algorithms sensitive to feature magnitudes.
"""
from typing import Any, Callable, Dict, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
import lightgbm as lgb


# Default hyperparameter configurations for UI expanders and factory instantiation
DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "Decision Tree": {
        "max_depth": 6,
        "min_samples_split": 2,
        "criterion": "gini",
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 2,
    },
    "Extra Trees": {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 2,
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
    },
    "LightGBM": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
    },
    "K-Nearest Neighbors (KNN)": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
    },
    "Support Vector Machine (SVM)": {
        "C": 1.0,
        "kernel": "rbf",
    },
    "MLP Neural Network": {
        "hidden_layer_sizes": (64, 32),
        "learning_rate_init": 0.001,
        "max_iter": 500,
    },
}

# Models that require feature scaling (standardization)
SCALED_MODELS = {
    "Logistic Regression",
    "K-Nearest Neighbors (KNN)",
    "Support Vector Machine (SVM)",
    "MLP Neural Network",
}


def _create_logistic_regression(task_type: str, **params: Any) -> Any:
    """Factory for Logistic Regression (Classifier) or Ridge (Regressor)."""
    if task_type == "regression":
        alpha = 1.0 / params.get("C", 1.0)
        base = Ridge(alpha=alpha, random_state=42)
    else:
        c_val = params.get("C", 1.0)
        max_iter = params.get("max_iter", 1000)
        solver = params.get("solver", "lbfgs")
        base = LogisticRegression(
            C=c_val,
            max_iter=max_iter,
            solver=solver,
            random_state=42,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", base)])


def _create_decision_tree(task_type: str, **params: Any) -> Any:
    """Factory for Decision Tree Classifier or Regressor."""
    max_depth = params.get("max_depth", 6)
    min_samples_split = params.get("min_samples_split", 2)
    criterion = params.get("criterion", "gini" if task_type == "classification" else "squared_error")

    if task_type == "regression":
        if criterion not in {"squared_error", "friedman_mse", "absolute_error", "poisson"}:
            criterion = "squared_error"
        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=42,
        )
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42,
    )


def _create_random_forest(task_type: str, **params: Any) -> Any:
    """Factory for Random Forest Classifier or Regressor."""
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 6)
    min_samples_split = params.get("min_samples_split", 2)

    if task_type == "regression":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1,
    )


def _create_extra_trees(task_type: str, **params: Any) -> Any:
    """Factory for Extra Trees Classifier or Regressor."""
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 6)
    min_samples_split = params.get("min_samples_split", 2)

    if task_type == "regression":
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
        )
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1,
    )


def _create_gradient_boosting(task_type: str, **params: Any) -> Any:
    """Factory for Gradient Boosting Classifier or Regressor."""
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 3)
    learning_rate = params.get("learning_rate", 0.1)

    if task_type == "regression":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
    )


def _create_xgboost(task_type: str, **params: Any) -> Any:
    """Factory for XGBoost Classifier or Regressor with existing default behavior preserved."""
    n_estimators = params.get("n_estimators", 200)
    max_depth = params.get("max_depth", 6)
    learning_rate = params.get("learning_rate", 0.1)

    if task_type == "regression":
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            random_state=42,
        )
    return xgb.XGBClassifier(
        objective="multi:softprob",
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="mlogloss",
        random_state=42,
    )


def _create_lightgbm(task_type: str, **params: Any) -> Any:
    """Factory for LightGBM Classifier or Regressor."""
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", 6)
    learning_rate = params.get("learning_rate", 0.1)
    num_leaves = params.get("num_leaves", 31)

    if task_type == "regression":
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=42,
            verbose=-1,
        )
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=42,
        verbose=-1,
    )


def _create_knn(task_type: str, **params: Any) -> Any:
    """Factory for K-Nearest Neighbors Classifier or Regressor wrapped in StandardScaler."""
    n_neighbors = params.get("n_neighbors", 5)
    weights = params.get("weights", "uniform")
    metric = params.get("metric", "minkowski")

    if task_type == "regression":
        base = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
        )
    else:
        base = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", base)])


def _create_svm(task_type: str, **params: Any) -> Any:
    """Factory for Support Vector Machine with probability support and StandardScaler."""
    c_val = params.get("C", 1.0)
    kernel = params.get("kernel", "rbf")

    if task_type == "regression":
        base = SVR(C=c_val, kernel=kernel)
    else:
        base = SVC(
            C=c_val,
            kernel=kernel,
            probability=True,
            random_state=42,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", base)])


def _create_mlp(task_type: str, **params: Any) -> Any:
    """Factory for MLP Neural Network with StandardScaler."""
    hidden_layer_sizes = params.get("hidden_layer_sizes", (64, 32))
    learning_rate_init = params.get("learning_rate_init", 0.001)
    max_iter = params.get("max_iter", 500)

    if task_type == "regression":
        base = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=42,
        )
    else:
        base = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=42,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", base)])


MODEL_REGISTRY: Dict[str, Callable[[str, Any], Any]] = {
    "Logistic Regression": _create_logistic_regression,
    "Decision Tree": _create_decision_tree,
    "Random Forest": _create_random_forest,
    "Extra Trees": _create_extra_trees,
    "Gradient Boosting": _create_gradient_boosting,
    "XGBoost": _create_xgboost,
    "LightGBM": _create_lightgbm,
    "K-Nearest Neighbors (KNN)": _create_knn,
    "Support Vector Machine (SVM)": _create_svm,
    "MLP Neural Network": _create_mlp,
}


def get_model(
    name: str,
    task_type: str = "classification",
    **hyperparams: Any,
) -> Any:
    """
    Retrieve and instantiate an estimator by display name from the registry.

    Args:
        name: Name of the model (must be in MODEL_REGISTRY).
        task_type: Either 'classification' or 'regression'. Defaults to 'classification'.
        **hyperparams: Optional keyword overrides for default hyperparameters.

    Returns:
        Instantiated scikit-learn estimator or Pipeline.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    # Merge defaults with custom overrides
    merged_params = dict(DEFAULT_HYPERPARAMS.get(name, {}))
    merged_params.update(hyperparams)

    factory = MODEL_REGISTRY[name]
    return factory(task_type, **merged_params)
