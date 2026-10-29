"""
Multimodal Fusion Engine for Data Lifecycle Management.
Supports both:
1. Early Fusion (feature-level concatenation across tabular, text, and graph modalities).
2. Late Fusion (probability stacking and weighted blending across modality-specific expert models).
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score

from src.evaluation.evaluate import FEATURE_COLUMNS, ACTION_SAVINGS_FACTOR, USD_TO_INR_RATE
from src.multimodal.text_encoder import TextSemanticsEncoder, extract_text_semantics
from src.multimodal.graph_encoder import GraphTopologyEncoder, load_or_create_dependency_graph
from src.training.models import get_model


class EarlyFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Early Fusion: Concatenates tabular features, dense text semantics embeddings,
    and structural graph embeddings into a unified feature vector, then fits a single classifier.
    """
    def __init__(
        self,
        base_model_name: str = "LightGBM",
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.base_model_name = base_model_name
        self.model_params = model_params or {}
        self.model = None
        self.is_fitted = False

    def fit(self, X_fused: np.ndarray, y: np.ndarray) -> "EarlyFusionClassifier":
        self.model = get_model(self.base_model_name, task_type="classification", **self.model_params)
        self.model.fit(X_fused, y)
        self.is_fitted = True
        return self

    def predict(self, X_fused: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("EarlyFusionClassifier must be fitted before predict().")
        return self.model.predict(X_fused)

    def predict_proba(self, X_fused: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("EarlyFusionClassifier must be fitted before predict_proba().")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_fused)
        # Fallback if model has no predict_proba: one-hot proxy
        preds = self.predict(X_fused)
        n_classes = len(np.unique(preds))
        proba = np.zeros((len(preds), n_classes), dtype=np.float32)
        for i, p in enumerate(preds):
            proba[i, p] = 1.0
        return proba

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            coef = np.array(self.model.coef_)
            return np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        return None


class LateFusionClassifier(BaseEstimator, ClassifierMixin):
    """
    Late Fusion: Fits 3 separate expert models on each distinct modality:
    - Tabular Expert
    - Text Semantics Expert
    - Graph Topology Expert
    Then combines their class probability distributions via meta-classifier probability stacking
    or weighted ensemble blending.
    """
    def __init__(
        self,
        tab_model_name: str = "LightGBM",
        text_model_name: str = "Random Forest",
        graph_model_name: str = "Logistic Regression",
        modality_weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
        use_stacking_meta: bool = True,
    ):
        self.tab_model_name = tab_model_name
        self.text_model_name = text_model_name
        self.graph_model_name = graph_model_name
        self.modality_weights = modality_weights
        self.use_stacking_meta = use_stacking_meta

        self.tab_model = None
        self.text_model = None
        self.graph_model = None
        self.meta_classifier = None
        self.classes_ = None
        self.is_fitted = False

    def fit(
        self,
        X_tab: np.ndarray,
        X_text: np.ndarray,
        X_graph: np.ndarray,
        y: np.ndarray,
    ) -> "LateFusionClassifier":
        self.classes_ = np.unique(y)

        # 1. Train Modality Experts
        self.tab_model = get_model(self.tab_model_name, task_type="classification")
        self.tab_model.fit(X_tab, y)

        self.text_model = get_model(self.text_model_name, task_type="classification")
        self.text_model.fit(X_text, y)

        self.graph_model = get_model(self.graph_model_name, task_type="classification")
        self.graph_model.fit(X_graph, y)

        # 2. Extract probability predictions
        p_tab = self._safe_predict_proba(self.tab_model, X_tab)
        p_text = self._safe_predict_proba(self.text_model, X_text)
        p_graph = self._safe_predict_proba(self.graph_model, X_graph)

        # 3. Train Meta-Classifier (Stacking) on concatenated probabilities
        stacked_features = np.hstack([p_tab, p_text, p_graph]).astype(np.float32)
        if self.use_stacking_meta:
            self.meta_classifier = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            self.meta_classifier.fit(stacked_features, y)

        self.is_fitted = True
        return self

    def _safe_predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # Ensure shape matches classes_
            if probs.shape[1] == len(self.classes_):
                return probs.astype(np.float32)

        preds = model.predict(X)
        one_hot = np.zeros((len(preds), len(self.classes_)), dtype=np.float32)
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0]
            if len(idx) > 0:
                one_hot[i, idx[0]] = 1.0
        return one_hot

    def predict_proba(
        self,
        X_tab: np.ndarray,
        X_text: np.ndarray,
        X_graph: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LateFusionClassifier must be fitted before predict_proba().")

        p_tab = self._safe_predict_proba(self.tab_model, X_tab)
        p_text = self._safe_predict_proba(self.text_model, X_text)
        p_graph = self._safe_predict_proba(self.graph_model, X_graph)

        if self.use_stacking_meta and self.meta_classifier is not None:
            stacked = np.hstack([p_tab, p_text, p_graph]).astype(np.float32)
            return self.meta_classifier.predict_proba(stacked).astype(np.float32)
        else:
            # Weighted blending
            w1, w2, w3 = self.modality_weights
            total_w = w1 + w2 + w3 if (w1 + w2 + w3) > 0 else 1.0
            blended = (w1 * p_tab + w2 * p_text + w3 * p_graph) / total_w
            return blended.astype(np.float32)

    def predict(
        self,
        X_tab: np.ndarray,
        X_text: np.ndarray,
        X_graph: np.ndarray,
    ) -> np.ndarray:
        probs = self.predict_proba(X_tab, X_text, X_graph)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]

    def get_modality_predictions(
        self,
        X_tab: np.ndarray,
        X_text: np.ndarray,
        X_graph: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Returns predictions and probabilities from each individual modality expert."""
        p_tab = self._safe_predict_proba(self.tab_model, X_tab)
        p_text = self._safe_predict_proba(self.text_model, X_text)
        p_graph = self._safe_predict_proba(self.graph_model, X_graph)

        return {
            "tabular": (self.classes_[np.argmax(p_tab, axis=1)], p_tab),
            "text": (self.classes_[np.argmax(p_text, axis=1)], p_text),
            "graph": (self.classes_[np.argmax(p_graph, axis=1)], p_graph),
        }


class MultimodalPipeline:
    """
    End-to-End Multimodal Pipeline Coordinator.
    Manages text extraction, graph topological encoding, tabular normalization,
    feature fusion (early or late), training, serialization, and real-time inference.
    """
    def __init__(
        self,
        fusion_type: str = "early",
        base_model_name: str = "LightGBM",
        text_dim: int = 16,
        graph_spectral_dim: int = 4,
        modality_weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
    ):
        self.fusion_type = fusion_type.lower()
        self.base_model_name = base_model_name
        self.text_dim = text_dim
        self.graph_spectral_dim = graph_spectral_dim
        self.modality_weights = modality_weights

        self.text_encoder = TextSemanticsEncoder(n_components=self.text_dim)
        self.graph_encoder = GraphTopologyEncoder(spectral_dim=self.graph_spectral_dim)
        self.tabular_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.early_classifier: Optional[EarlyFusionClassifier] = None
        self.late_classifier: Optional[LateFusionClassifier] = None
        self.is_fitted = False

    def extract_texts(self, df: pd.DataFrame) -> List[str]:
        """Extracts code comments and semantic tokens for each row in df."""
        fpaths = df["full_path"].tolist() if "full_path" in df.columns else [""] * len(df)
        rpaths = df["relative_path"].tolist() if "relative_path" in df.columns else [""] * len(df)
        return [extract_text_semantics(filepath=fp, relative_path=rp) for fp, rp in zip(fpaths, rpaths)]

    def fit(
        self,
        train_df: pd.DataFrame,
        G: Optional[Any] = None,
        label_col: str = "lifecycle_label",
    ) -> "MultimodalPipeline":
        """Fits encoders, scalers, and fusion models on train_df and dependency graph G."""
        # 1. Fit LabelEncoder
        y = self.label_encoder.fit_transform(train_df[label_col])

        # 2. Extract & Encode Text Semantics
        texts = self.extract_texts(train_df)
        X_text = self.text_encoder.fit_transform(texts)

        # 3. Extract & Encode Graph Topology
        object_ids = train_df["object_id"].tolist() if "object_id" in train_df.columns else [f"obj_{i}" for i in range(len(train_df))]
        if G is None:
            G = load_or_create_dependency_graph(df=train_df)
        self.graph_encoder.fit(G, object_ids=object_ids)
        X_graph = self.graph_encoder.transform(object_ids, fallback_df=train_df)

        # 4. Tabular Features
        X_tab_raw = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        X_tab = self.tabular_scaler.fit_transform(X_tab_raw).astype(np.float32)

        # 5. Fit Fusion Model
        if self.fusion_type == "early":
            X_fused = np.hstack([X_tab, X_text, X_graph]).astype(np.float32)
            self.early_classifier = EarlyFusionClassifier(base_model_name=self.base_model_name)
            self.early_classifier.fit(X_fused, y)
        else:
            self.late_classifier = LateFusionClassifier(
                tab_model_name="LightGBM",
                text_model_name="Random Forest",
                graph_model_name="Logistic Regression",
                modality_weights=self.modality_weights,
            )
            self.late_classifier.fit(X_tab, X_text, X_graph, y)

        self.is_fitted = True
        return self

    def transform_modalities(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encodes tabular, text, and graph modalities for df."""
        if not self.is_fitted:
            raise RuntimeError("MultimodalPipeline must be fitted first.")

        # Tabular
        X_tab_raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        X_tab = self.tabular_scaler.transform(X_tab_raw).astype(np.float32)

        # Text
        texts = self.extract_texts(df)
        X_text = self.text_encoder.transform(texts)

        # Graph
        object_ids = df["object_id"].tolist() if "object_id" in df.columns else [f"obj_{i}" for i in range(len(df))]
        X_graph = self.graph_encoder.transform(object_ids, fallback_df=df)

        return X_tab, X_text, X_graph

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts policy action strings (retain, compress, dedup, archive, delete)."""
        X_tab, X_text, X_graph = self.transform_modalities(df)

        if self.fusion_type == "early":
            X_fused = np.hstack([X_tab, X_text, X_graph]).astype(np.float32)
            y_pred_idx = self.early_classifier.predict(X_fused)
        else:
            y_pred_idx = self.late_classifier.predict(X_tab, X_text, X_graph)

        return self.label_encoder.inverse_transform(y_pred_idx)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities across all 5 actions."""
        X_tab, X_text, X_graph = self.transform_modalities(df)

        if self.fusion_type == "early":
            X_fused = np.hstack([X_tab, X_text, X_graph]).astype(np.float32)
            return self.early_classifier.predict_proba(X_fused)
        else:
            return self.late_classifier.predict_proba(X_tab, X_text, X_graph)

    def predict_single(
        self,
        tabular_dict: Dict[str, Any],
        source_code_or_comment: str = "",
        graph_dict: Optional[Dict[str, Any]] = None,
        relative_path: str = "module_sim.c",
    ) -> Dict[str, Any]:
        """
        Interactive live inference method for single-object simulation.
        Ingests user inputs across all 3 modalities and computes unified predictions,
        probabilities, and per-modality consensus votes.
        """
        full_tab = {
            "redundancy_score": 0.0,
            "object_size": 65536,
            "object_age_days": 30,
            "access_freq_7d": 0,
            "access_freq_30d": 0,
            "access_freq_90d": 0,
            "days_since_last_access": 0,
            "num_dependents": 0,
            "compression_ratio": 2.0,
            "compute_time_sec": 0.01,
            "monthly_storage_cost_usd": 0.001,
        }
        full_tab.update(tabular_dict)
        tab_df = pd.DataFrame([full_tab])
        for col in FEATURE_COLUMNS:
            if col not in tab_df.columns:
                tab_df[col] = 0.0

        X_tab_raw = tab_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        X_tab = self.tabular_scaler.transform(X_tab_raw).astype(np.float32)

        # Text encoding
        combined_text = extract_text_semantics(
            filepath=None,
            relative_path=relative_path,
            fallback_text=source_code_or_comment,
        )
        X_text = self.text_encoder.transform([combined_text])

        # Graph encoding
        g_in = float(graph_dict.get("in_degree", tabular_dict.get("num_dependents", 0))) if graph_dict else float(tabular_dict.get("num_dependents", 0))
        g_out = float(graph_dict.get("out_degree", 1.0)) if graph_dict else 1.0
        ratio = g_in / (g_in + g_out + 1.0)
        pr = float(graph_dict.get("pagerank", 0.01)) if graph_dict else 0.01
        clust = float(graph_dict.get("clustering", 0.1)) if graph_dict else 0.1
        core = float(graph_dict.get("core_number", 1)) if graph_dict else 1.0

        raw_graph_metrics = np.array([g_in, g_out, ratio, pr, clust, core], dtype=np.float32)
        spec_zeros = np.zeros(self.graph_spectral_dim, dtype=np.float32)
        raw_graph_vec = np.hstack([raw_graph_metrics, spec_zeros])
        X_graph = self.graph_encoder.scaler.transform([raw_graph_vec]).astype(np.float32)

        # Compute Early prediction
        X_fused = np.hstack([X_tab, X_text, X_graph]).astype(np.float32)
        if self.early_classifier is not None:
            early_proba = self.early_classifier.predict_proba(X_fused)[0]
            early_action = self.label_encoder.inverse_transform([np.argmax(early_proba)])[0]
        else:
            early_proba = np.ones(len(self.label_encoder.classes_)) / len(self.label_encoder.classes_)
            early_action = "retain"

        # Compute Late prediction and modality expert breakdown
        modality_breakdown = {}
        if self.late_classifier is not None:
            late_proba = self.late_classifier.predict_proba(X_tab, X_text, X_graph)[0]
            late_action = self.label_encoder.inverse_transform([np.argmax(late_proba)])[0]
            sub_preds = self.late_classifier.get_modality_predictions(X_tab, X_text, X_graph)
            for m_key, (pred_idx, p_arr) in sub_preds.items():
                m_action = self.label_encoder.inverse_transform(pred_idx)[0]
                m_conf = float(np.max(p_arr[0]))
                modality_breakdown[m_key] = {"action": m_action, "confidence": m_conf, "probs": p_arr[0].tolist()}
        else:
            late_proba = early_proba
            late_action = early_action
            modality_breakdown = {
                "tabular": {"action": early_action, "confidence": float(np.max(early_proba)), "probs": early_proba.tolist()},
                "text": {"action": early_action, "confidence": float(np.max(early_proba)), "probs": early_proba.tolist()},
                "graph": {"action": early_action, "confidence": float(np.max(early_proba)), "probs": early_proba.tolist()},
            }

        # Active prediction based on current pipeline fusion_type
        active_action = early_action if self.fusion_type == "early" else late_action
        active_proba = early_proba if self.fusion_type == "early" else late_proba
        savings_factor = ACTION_SAVINGS_FACTOR.get(active_action, 0.0)

        return {
            "action": active_action,
            "confidence": float(np.max(active_proba)),
            "probabilities": {cls_name: float(active_proba[idx]) for idx, cls_name in enumerate(self.label_encoder.classes_)},
            "savings_factor": savings_factor,
            "early_action": early_action,
            "early_confidence": float(np.max(early_proba)),
            "late_action": late_action,
            "late_confidence": float(np.max(late_proba)),
            "modality_breakdown": modality_breakdown,
            "fused_vector_dim": X_fused.shape[1],
        }

    @property
    def feature_names_fused(self) -> List[str]:
        """Returns names of all concatenated features in early fusion."""
        return (
            [f"tab_{col}" for col in FEATURE_COLUMNS]
            + self.text_encoder.feature_names
            + self.graph_encoder.feature_names
        )

    def save(self, filepath: str = "models/multimodal_pipeline.pkl") -> None:
        """Saves pipeline state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/multimodal_pipeline.pkl") -> "MultimodalPipeline":
        """Loads serialized pipeline state."""
        return joblib.load(filepath)
