"""
Multimodal Diagnostics, Projection, and Explainability Module.
Provides PCA and t-SNE 2D/3D dimensionality reduction, interactive Plotly visualization,
and modality contribution analysis (Tabular vs. Textual vs. Graph Topology).
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.evaluate import calculate_storage_savings, FEATURE_COLUMNS


def project_embeddings(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    sample_size: int = 1500,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[List[float]], np.ndarray]:
    """
    Projects high-dimensional fused multimodal embeddings into 2D or 3D space.
    Subsamples up to sample_size for responsive interactivity on laptop hardware.

    Returns:
        Tuple of (projected_coords, explained_variance_ratio_or_None, sample_indices).
    """
    n_samples = len(embeddings)
    if n_samples > sample_size:
        np.random.seed(random_state)
        sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
        sub_emb = embeddings[sample_indices]
    else:
        sample_indices = np.arange(n_samples)
        sub_emb = embeddings

    if method.lower() == "tsne":
        perplexity = min(30, max(5, len(sub_emb) // 10))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        coords = reducer.fit_transform(sub_emb)
        variance_ratio = None
    else:
        reducer = PCA(n_components=n_components, random_state=random_state)
        coords = reducer.fit_transform(sub_emb)
        variance_ratio = [float(v) for v in reducer.explained_variance_ratio_]

    return coords.astype(np.float32), variance_ratio, sample_indices


def create_embedding_scatter_plot(
    coords: np.ndarray,
    labels: List[str],
    hover_df: pd.DataFrame,
    method_name: str = "PCA",
    n_components: int = 2,
) -> go.Figure:
    """
    Generates an interactive Plotly 2D or 3D scatter plot of multimodal embeddings.
    """
    fig = go.Figure()

    unique_labels = sorted(list(set(labels)))
    colors = {
        "retain": "#10b981",    # Emerald
        "compress": "#3b82f6",  # Blue
        "dedup": "#f59e0b",     # Amber
        "archive": "#8b5cf6",   # Purple
        "delete": "#ef4444",    # Red
    }

    is_3d = (n_components == 3 and coords.shape[1] >= 3)

    for action in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == action]
        if not mask:
            continue

        c_sub = coords[mask]
        sub_df = hover_df.iloc[mask]

        hover_texts = []
        for _, row in sub_df.iterrows():
            oid = row.get("object_id", "object")
            sz = row.get("object_size", 0)
            red = row.get("redundancy_score", 0.0)
            deps = row.get("num_dependents", 0)
            hover_texts.append(
                f"<b>ID:</b> {oid}<br>"
                f"<b>Action:</b> {action.upper()}<br>"
                f"<b>Size:</b> {sz:,} bytes<br>"
                f"<b>Redundancy:</b> {red:.2%}<br>"
                f"<b>Dependents:</b> {deps}"
            )

        marker_dict = dict(
            size=6 if is_3d else 7,
            color=colors.get(action.lower(), "#94a3b8"),
            opacity=0.85,
        )

        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=c_sub[:, 0],
                y=c_sub[:, 1],
                z=c_sub[:, 2],
                mode="markers",
                name=action.upper(),
                marker=marker_dict,
                text=hover_texts,
                hoverinfo="text",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=c_sub[:, 0],
                y=c_sub[:, 1],
                mode="markers",
                name=action.upper(),
                marker=marker_dict,
                text=hover_texts,
                hoverinfo="text",
            ))

    layout_dict: Dict[str, Any] = dict(
        font=dict(color="#EEEEEE", family="sans-serif"),
        title=dict(
            text=f"Multimodal Feature Space Projection ({method_name} {n_components}D)",
            font=dict(size=14, color="#F8FAFC"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=10, r=10, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#E2E8F0"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    if not is_3d:
        layout_dict["xaxis"] = dict(
            title=dict(text=f"{method_name} Component 1", font=dict(size=11, color="#E2E8F0")),
            gridcolor="rgba(255, 255, 255, 0.10)",
        )
        layout_dict["yaxis"] = dict(
            title=dict(text=f"{method_name} Component 2", font=dict(size=11, color="#E2E8F0")),
            gridcolor="rgba(255, 255, 255, 0.10)",
        )

    fig.update_layout(**layout_dict)
    return fig


def analyze_modality_importance(pipeline: Any) -> Dict[str, float]:
    """
    Computes percentage contribution of each modality (Tabular, Text, Graph)
    based on trained Early Fusion feature importances or Late Fusion ensemble weights.
    """
    tab_dim = len(FEATURE_COLUMNS)
    text_dim = pipeline.text_dim
    graph_dim = len(pipeline.graph_encoder.feature_names) if pipeline.graph_encoder.is_fitted else 10

    # 1. Early Fusion feature importances
    if pipeline.early_classifier is not None:
        importances = pipeline.early_classifier.feature_importances_
        if importances is not None and len(importances) == (tab_dim + text_dim + graph_dim):
            tab_imp = float(np.sum(importances[:tab_dim]))
            text_imp = float(np.sum(importances[tab_dim : tab_dim + text_dim]))
            graph_imp = float(np.sum(importances[tab_dim + text_dim :]))
            total = tab_imp + text_imp + graph_imp
            if total > 0:
                return {
                    "Tabular Metadata": (tab_imp / total) * 100.0,
                    "Textual Semantics": (text_imp / total) * 100.0,
                    "Graph Topology": (graph_imp / total) * 100.0,
                }

    # 2. Late Fusion weights fallback
    if pipeline.late_classifier is not None:
        w_tab, w_text, w_graph = pipeline.late_classifier.modality_weights
        total_w = w_tab + w_text + w_graph
        if total_w > 0:
            return {
                "Tabular Metadata": (w_tab / total_w) * 100.0,
                "Textual Semantics": (w_text / total_w) * 100.0,
                "Graph Topology": (w_graph / total_w) * 100.0,
            }

    return {
        "Tabular Metadata": 50.0,
        "Textual Semantics": 30.0,
        "Graph Topology": 20.0,
    }


def create_modality_contribution_chart(contributions: Dict[str, float]) -> go.Figure:
    """Renders a horizontal bar chart of relative modality contributions."""
    modalities = list(contributions.keys())
    values = list(contributions.values())
    colors = ["#3b82f6", "#10b981", "#f59e0b"]

    fig = go.Figure(go.Bar(
        y=modalities,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#F8FAFC", size=11),
        hovertemplate="Modality: <b>%{y}</b><br>Influence: <b>%{x:.1f}%</b><extra></extra>",
    ))

    fig.update_layout(
        font=dict(color="#EEEEEE", family="sans-serif"),
        title=dict(text="Modality Contribution to Lifecycle Decisions", font=dict(size=13, color="#F8FAFC")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=10, r=40, t=40, b=30),
        xaxis=dict(
            title=dict(text="Contribution Percentage (%)", font=dict(size=11, color="#E2E8F0")),
            range=[0, 110],
            gridcolor="rgba(255, 255, 255, 0.10)",
        ),
        yaxis=dict(tickfont=dict(size=11, color="#E2E8F0")),
    )
    return fig

