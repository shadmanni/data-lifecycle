"""
Multimodal Feature Integration Package for Cloud Data Lifecycle Management.
Fuses Tabular Metadata, Textual Semantics, and Structural Graph Topology.
"""
from src.multimodal.text_encoder import TextSemanticsEncoder, extract_text_semantics
from src.multimodal.graph_encoder import GraphTopologyEncoder, load_or_create_dependency_graph
from src.multimodal.fusion import (
    EarlyFusionClassifier,
    LateFusionClassifier,
    MultimodalPipeline,
)
from src.multimodal.diagnostics import (
    project_embeddings,
    create_embedding_scatter_plot,
    analyze_modality_importance,
    create_modality_contribution_chart,
)

__all__ = [
    "TextSemanticsEncoder",
    "extract_text_semantics",
    "GraphTopologyEncoder",
    "load_or_create_dependency_graph",
    "EarlyFusionClassifier",
    "LateFusionClassifier",
    "MultimodalPipeline",
    "project_embeddings",
    "create_embedding_scatter_plot",
    "analyze_modality_importance",
    "create_modality_contribution_chart",
]

