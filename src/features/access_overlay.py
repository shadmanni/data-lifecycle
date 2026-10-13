"""
Overlays realistic access-frequency and recency features onto each object.
Uses a Zipf distribution, which is the standard, well-documented shape of
real-world storage access skew (a small fraction of objects get most accesses).
Memory-optimized with downcasting to 16-bit integers and 32-bit floats.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.memory_opt import optimize_dtypes, MemoryTracker


def overlay_access_patterns(
    df: pd.DataFrame,
    seed: int = 42,
    zipf_param: float = 1.2,
    days_window: int = 180,
) -> pd.DataFrame:
    with MemoryTracker("Overlay Access Patterns"):
        rng = np.random.default_rng(seed)
        n = len(df)

        ranks = rng.permutation(n) + 1
        weights = 1.0 / (ranks.astype(np.float64) ** zipf_param)
        weights = weights / weights.sum()

        total_events = n * 15  # tune this multiplier to control overall access density
        access_counts_90d = rng.multinomial(total_events, weights).astype(np.uint32)

        # 30-day and 7-day counts are subsets of the 90-day count, weighted toward recency
        access_counts_30d = np.array([rng.binomial(c, 0.5) for c in access_counts_90d], dtype=np.uint32)
        access_counts_7d = np.array([rng.binomial(c, 0.3) for c in access_counts_30d], dtype=np.uint32)

        # days since last access: objects with more accesses tend to have been accessed more recently
        days_since_last = np.array([
            rng.integers(0, days_window) if c == 0 else rng.integers(0, max(1, days_window // (c + 1)))
            for c in access_counts_90d
        ], dtype=np.uint16)

        # synthetic object age in days
        object_age_days = rng.integers(1, 1000, size=n, dtype=np.uint16)

        df = df.copy()
        df["access_freq_7d"] = access_counts_7d
        df["access_freq_30d"] = access_counts_30d
        df["access_freq_90d"] = access_counts_90d.astype(np.uint16)
        df["days_since_last_access"] = days_since_last
        df["object_age_days"] = object_age_days

        df = optimize_dtypes(df, verbose=True)
    return df


if __name__ == "__main__":
    in_path = os.path.join("data", "processed", "objects_with_redundancy.parquet")
    df = pd.read_parquet(in_path)
    df = overlay_access_patterns(df)
    out_path = os.path.join("data", "processed", "objects_with_access.parquet")
    df.to_parquet(out_path, compression="snappy", index=False)
    print(df[["access_freq_7d", "access_freq_30d", "access_freq_90d", "days_since_last_access"]].describe())
    print(f"Saved to {out_path}")