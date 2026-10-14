"""
Overlays realistic access-frequency and recency features onto each object.
Uses real Azure Functions 2019 trace data if available, with a Zipf distribution fallback.
"""
import os
import numpy as np
import pandas as pd

AZURE_DIR = os.path.join("data", "raw", "azure_functions_2019")


def load_azure_invocation_distribution(azure_dir=AZURE_DIR, sample_size=5000):
    """Load empirical invocation counts from Azure Functions 2019 trace."""
    duration_file = os.path.join(azure_dir, "function_durations_percentiles.anon.d01.csv")
    if os.path.isfile(duration_file):
        try:
            # Read 'Count' column which represents daily function invocations
            df_trace = pd.read_csv(duration_file, usecols=["Count"], nrows=sample_size)
            counts = df_trace["Count"].dropna().astype(int).values
            counts = counts[counts > 0]
            if len(counts) > 0:
                return counts
        except Exception as e:
            print(f"Notice: Could not parse Azure trace ({e}), using Zipf fallback.")
    return None


def overlay_access_patterns(df: pd.DataFrame, seed: int = 42, zipf_param: float = 1.2,
                            days_window: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)

    azure_counts = load_azure_invocation_distribution()
    if azure_counts is not None and len(azure_counts) > 0:
        # Sample empirical invocation rates from real Azure Functions production traces
        sampled_daily = rng.choice(azure_counts, size=n, replace=True)
        # Scale to 90d access counts, bounded reasonably
        access_counts_90d = np.clip((sampled_daily * rng.uniform(0.1, 1.5, size=n)).astype(int), 0, 500)
    else:
        ranks = rng.permutation(n) + 1
        weights = 1.0 / (ranks.astype(float) ** zipf_param)
        weights = weights / weights.sum()
        total_events = n * 15
        access_counts_90d = rng.multinomial(total_events, weights)

    # 30-day and 7-day counts are subsets of the 90-day count, weighted toward recency
    access_counts_30d = np.array([rng.binomial(c, 0.5) for c in access_counts_90d])
    access_counts_7d = np.array([rng.binomial(c, 0.3) for c in access_counts_30d])

    # days since last access: objects with more accesses tend to have been accessed more recently
    days_since_last = np.array([
        rng.integers(0, days_window) if c == 0 else rng.integers(0, max(1, days_window // (c + 1)))
        for c in access_counts_90d
    ])

    object_age_days = rng.integers(1, 1000, size=n)

    df = df.copy()
    df["access_freq_7d"] = access_counts_7d
    df["access_freq_30d"] = access_counts_30d
    df["access_freq_90d"] = access_counts_90d
    df["days_since_last_access"] = days_since_last
    df["object_age_days"] = object_age_days
    return df


if __name__ == "__main__":
    in_path = os.path.join("data", "processed", "objects_with_redundancy.parquet")
    df = pd.read_parquet(in_path)
    df = overlay_access_patterns(df)
    out_path = os.path.join("data", "processed", "objects_with_access.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(df[["access_freq_7d", "access_freq_30d", "access_freq_90d", "days_since_last_access"]].describe())
    print(f"Saved to {out_path}")