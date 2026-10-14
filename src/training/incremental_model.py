"""
Incremental model wrappers and pipelines for batch-wise partial_fit learning.
"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin


class IncrementalPipelineWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible incremental classification pipeline.
    Combines incremental standardization (StandardScaler.partial_fit)
    and incremental logistic regression (SGDClassifier.partial_fit).
    Provides predict() and predict_proba() for seamless downstream evaluation.
    """
    def __init__(self, classes=None):
        self.classes = classes
        self.classes_ = np.array(classes) if classes is not None else None
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1000,
            random_state=42,
        )
        self.is_fitted = False

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = np.array(classes)
        # Update scaler incrementally
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        self.clf.partial_fit(X_scaled, y, classes=self.classes_)
        self.is_fitted = True
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self.partial_fit(X, y, classes=self.classes_)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    @property
    def feature_importances_(self):
        if hasattr(self.clf, "coef_"):
            return np.mean(np.abs(self.clf.coef_), axis=0)
        return np.ones(11)
