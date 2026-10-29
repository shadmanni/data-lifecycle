"""
Textual Semantics Ingestion and Encoding Module.
Parses source code comments, docstrings, include headers, and README text,
and encodes them into compact, CPU-efficient dense embeddings using LSA (TF-IDF + TruncatedSVD).
"""
from typing import Dict, List, Optional, Tuple, Union
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Common programming and software lifecycle keywords for fallback generation
C_COMMENT_LINE_RE = re.compile(r"//.*")
C_COMMENT_BLOCK_RE = re.compile(r"/\*[\s\S]*?\*/")
PY_COMMENT_RE = re.compile(r"#.*")
PY_DOCSTRING_RE = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
INCLUDE_RE = re.compile(r'#include\s*[<"]([^>"]+)[>"]')
FUNCTION_DECL_RE = re.compile(r"(?:void|int|char|float|double|struct|static)\s+([a-zA-Z0-9_]+)\s*\(")


def extract_text_semantics(filepath: Optional[str] = None, relative_path: str = "", fallback_text: str = "") -> str:
    """
    Extracts semantic textual content from a code file, including comments, docstrings,
    include headers, and function names. If the file is not on disk, generates a rich
    semantic text representation from the filename, path, and metadata.
    """
    extracted_tokens: List[str] = []

    # 1. Try reading the file from disk if it exists
    if filepath and os.path.isfile(filepath):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(50000)  # Read up to first 50KB to maintain O(1) memory

            # Extract block comments and docstrings
            for block in C_COMMENT_BLOCK_RE.findall(content):
                clean_block = re.sub(r"/\*+|\*+/|\*", " ", block).strip()
                if clean_block:
                    extracted_tokens.append(clean_block)

            for doc in PY_DOCSTRING_RE.findall(content):
                clean_doc = doc.strip("\"' \n\t")
                if clean_doc:
                    extracted_tokens.append(clean_doc)

            # Extract line comments
            for line_comment in C_COMMENT_LINE_RE.findall(content):
                clean_c = line_comment.lstrip("/ ").strip()
                if clean_c:
                    extracted_tokens.append(clean_c)

            for py_comment in PY_COMMENT_RE.findall(content):
                clean_py = py_comment.lstrip("# ").strip()
                if clean_py:
                    extracted_tokens.append(clean_py)

            # Extract includes and headers
            for inc in INCLUDE_RE.findall(content):
                extracted_tokens.append(f"header {inc.replace('.', ' ')}")

            # Extract function declarations
            for fn in FUNCTION_DECL_RE.findall(content):
                extracted_tokens.append(f"function {fn.replace('_', ' ')}")

            # If no comments found, extract the first 30 lines of code text
            if not extracted_tokens:
                lines = [l.strip() for l in content.splitlines()[:30] if l.strip()]
                clean_code = " ".join(lines)
                clean_code = re.sub(r"[^a-zA-Z0-9_\s]", " ", clean_code)
                extracted_tokens.append(clean_code)

        except Exception:
            pass

    # 2. Enrich with path and fallback semantic tokens
    path_ref = relative_path or (os.path.basename(filepath) if filepath else "")
    if path_ref:
        base_name = os.path.basename(path_ref)
        name_no_ext, ext = os.path.splitext(base_name)
        clean_path_words = re.sub(r"[^a-zA-Z0-9]", " ", path_ref).lower()
        extracted_tokens.append(f"file {clean_path_words} extension {ext.replace('.', '')}")

        # Semantic keywords derived from file naming conventions
        if any(term in clean_path_words for term in ["test", "mock", "spec"]):
            extracted_tokens.append("test suite verification assertion mock")
        if any(term in clean_path_words for term in ["config", "setup", "init", "param"]):
            extracted_tokens.append("configuration initialization parameter settings")
        if any(term in clean_path_words for term in ["util", "helper", "tool", "common"]):
            extracted_tokens.append("utility helper shared common library routine")
        if any(term in clean_path_words for term in ["kernel", "driver", "sched", "core"]):
            extracted_tokens.append("core kernel subsystem low level critical driver")
        if any(term in clean_path_words for term in ["cache", "buffer", "queue", "pool"]):
            extracted_tokens.append("transient cache buffer memory queue pool")

    if fallback_text:
        extracted_tokens.append(fallback_text)

    combined = " ".join(extracted_tokens).strip()
    return combined if combined else "generic source code object module component"


class TextSemanticsEncoder(BaseEstimator, TransformerMixin):
    """
    Lightweight, CPU-only semantic text embedding transformer.
    Uses TF-IDF feature extraction followed by TruncatedSVD (Latent Semantic Analysis)
    and L2/Standard scaling to produce a dense embedding matrix.
    """
    def __init__(
        self,
        n_components: int = 16,
        max_features: int = 512,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, texts: List[str], y=None) -> "TextSemanticsEncoder":
        """Fits the TF-IDF vectorizer and SVD decomposition on a collection of texts."""
        if not texts or len(texts) == 0:
            texts = ["generic software module component source code"]

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        # Adapt n_components if vocabulary or samples are smaller than requested
        effective_components = min(self.n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        effective_components = max(2, effective_components)

        if effective_components != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=effective_components, random_state=self.random_state)

        svd_matrix = self.svd.fit_transform(tfidf_matrix)
        self.scaler.fit(svd_matrix)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforms a collection of texts into standardized dense embeddings."""
        if not self.is_fitted:
            raise RuntimeError("TextSemanticsEncoder must be fitted before calling transform().")

        if not texts or len(texts) == 0:
            texts = ["generic software module component source code"]

        tfidf_matrix = self.vectorizer.transform(texts)
        svd_matrix = self.svd.transform(tfidf_matrix)
        scaled_matrix = self.scaler.transform(svd_matrix)
        return scaled_matrix.astype(np.float32)

    def fit_transform(self, texts: List[str], y=None) -> np.ndarray:
        """Fits on texts and transforms them in one pass."""
        return self.fit(texts, y).transform(texts)

    @property
    def feature_names(self) -> List[str]:
        """Returns column names for the text embedding dimensions."""
        dim = self.svd.n_components if self.is_fitted else self.n_components
        return [f"text_emb_{i:02d}" for i in range(dim)]

    def get_top_terms_per_component(self, n_top_terms: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Returns the most influential vocabulary terms for each LSA component.
        Useful for UI explainability and diagnostics.
        """
        if not self.is_fitted or not hasattr(self.vectorizer, "get_feature_names_out"):
            return {}

        terms = self.vectorizer.get_feature_names_out()
        res = {}
        for comp_idx, comp in enumerate(self.svd.components_):
            top_indices = np.argsort(np.abs(comp))[::-1][:n_top_terms]
            res[f"text_emb_{comp_idx:02d}"] = [(str(terms[i]), float(comp[i])) for i in top_indices]
        return res

