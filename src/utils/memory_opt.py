"""
Memory optimization, downcasting, and memory tracking utilities
for running large-scale data lifecycle workflows on restricted laptop environments.
"""
import os
import gc
import logging
import psutil
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("memory_opt")


def get_memory_info():
    """Returns current process RSS memory in MB and total system RAM usage percentage."""
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    system_mem = psutil.virtual_memory()
    return {
        "process_rss_mb": rss_mb,
        "system_total_gb": system_mem.total / (1024 ** 3),
        "system_available_gb": system_mem.available / (1024 ** 3),
        "system_percent": system_mem.percent,
    }


def log_memory(stage: str = ""):
    """Logs current memory status."""
    info = get_memory_info()
    logger.info(
        f"[RAM Monitor] {stage} -> Process RSS: {info['process_rss_mb']:.1f} MB | "
        f"Sys Avail: {info['system_available_gb']:.2f} GB ({info['system_percent']}% used)"
    )


class MemoryTracker:
    """
    Context manager to track memory usage before and after a critical block.
    Raises warnings or triggers garbage collection if threshold is exceeded.
    """
    def __init__(self, stage_name: str = "Block", alert_threshold_percent: float = 85.0):
        self.stage_name = stage_name
        self.alert_threshold_percent = alert_threshold_percent
        self.start_mem = None

    def __enter__(self):
        gc.collect()
        self.start_mem = get_memory_info()
        logger.info(
            f"--> Starting '{self.stage_name}': Process RSS={self.start_mem['process_rss_mb']:.1f} MB "
            f"(System {self.start_mem['system_percent']}%)"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        end_mem = get_memory_info()
        delta = end_mem["process_rss_mb"] - self.start_mem["process_rss_mb"]
        logger.info(
            f"<-- Completed '{self.stage_name}': Process RSS={end_mem['process_rss_mb']:.1f} MB "
            f"(Delta: {delta:+.1f} MB, System {end_mem['system_percent']}%)"
        )
        if end_mem["system_percent"] >= self.alert_threshold_percent:
            logger.warning(
                f"HIGH MEMORY ALERT in '{self.stage_name}': System RAM is {end_mem['system_percent']}%! "
                f"Triggering emergency GC collection."
            )
            gc.collect()


def optimize_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Downcasts numeric and categorical columns to their smallest safe representation.
    Reduces pandas DataFrame memory footprint by 50-75%.
    """
    start_bytes = df.memory_usage(deep=True).sum()
    df = df.copy()

    for col in df.columns:
        col_type = df[col].dtype

        # Already categorical -> skip
        if isinstance(col_type, pd.CategoricalDtype) or str(col_type) == "category":
            continue

        # Handle Categorical / String downcasting for low cardinality
        if col_type == object or isinstance(col_type, pd.StringDtype):
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_val, (list, np.ndarray, set, dict)):
                continue  # skip list/array columns like chunk_hashes
            try:
                num_unique = df[col].nunique(dropna=False)
                num_total = len(df[col])
                if num_total > 0 and (num_unique / num_total < 0.2 or col in ["version", "lifecycle_label"]):
                    df[col] = df[col].astype("category")
            except (TypeError, ValueError):
                continue

        # Downcast Integer columns
        elif pd.api.types.is_integer_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isna(c_min) or pd.isna(c_max):
                continue
            if c_min >= 0:
                if c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

        # Downcast Float columns (float64 -> float32)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)

    end_bytes = df.memory_usage(deep=True).sum()
    if verbose and start_bytes > 0:
        saved_pct = 100 * (1 - end_bytes / start_bytes)
        logger.info(
            f"Downcasted DataFrame: {start_bytes / (1024**2):.2f} MB -> "
            f"{end_bytes / (1024**2):.2f} MB ({saved_pct:.1f}% reduction)"
        )
    return df


def validate_batch_integrity(df: pd.DataFrame, feature_cols: list, target_col: str = None):
    """
    Automated assertion checks for data batch integrity:
    1. Checks for NaN or Infinite values
    2. Checks feature column ranges
    3. Validates target distribution if target_col is provided
    """
    assert not df.empty, "DataFrame batch is unexpectedly empty!"

    # 1. Check feature columns presence
    missing_cols = [c for c in feature_cols if c not in df.columns]
    assert not missing_cols, f"Missing required feature columns in batch: {missing_cols}"

    # 2. Check NaNs
    nan_counts = df[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    assert len(nan_cols) == 0, f"NaN values detected in batch features: {nan_cols.to_dict()}"

    # 3. Check Infs
    for col in feature_cols:
        if np.issubdtype(df[col].dtype, np.number):
            has_inf = np.isinf(df[col]).any()
            assert not has_inf, f"Infinite values detected in column '{col}'"

    # 4. Check Value Ranges
    if "redundancy_score" in df.columns:
        assert (df["redundancy_score"] >= -1e-5).all() and (df["redundancy_score"] <= 1.0 + 1e-5).all(), \
            "redundancy_score values must be bounded in [0.0, 1.0]"

    if "object_size" in df.columns:
        assert (df["object_size"] >= 0).all(), "object_size cannot be negative"

    if "monthly_storage_cost_usd" in df.columns:
        assert (df["monthly_storage_cost_usd"] >= 0).all(), "monthly_storage_cost_usd cannot be negative"

    # 5. Check Target
    if target_col and target_col in df.columns:
        assert not df[target_col].isna().any(), f"Target column '{target_col}' contains NaN values!"
        unique_targets = df[target_col].nunique()
        assert unique_targets > 0, "No valid targets found in batch"

    return True
