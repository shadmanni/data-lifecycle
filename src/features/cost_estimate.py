"""
Estimates compression benefit and compute cost per object using zlib
(built into Python, works identically on every OS -- avoids xdelta/Windows issues).
This is a PROXY for true delta compression, documented as such in the paper.
"""
import os
import time
import zlib
import pandas as pd

# Approximate public cloud storage tier pricing (USD per GB-month).
# Verify current numbers against AWS/Azure pricing pages before publishing exact figures.
STORAGE_COST_PER_GB_MONTH = {
    "hot": 0.023,
    "cool": 0.0125,
    "cold": 0.004,
    "archive": 0.00099,
}


def estimate_cost(filepath: str):
    try:
        with open(filepath, "rb") as f:
            data = f.read()
    except Exception:
        return None

    start = time.perf_counter()
    compressed = zlib.compress(data, level=6)
    elapsed = time.perf_counter() - start

    original_size = len(data)
    compressed_size = len(compressed)
    ratio = original_size / max(1, compressed_size)

    return {
        "compression_ratio": ratio,
        "compute_time_sec": elapsed,
        "compressed_size_bytes": compressed_size,
    }


if __name__ == "__main__":
    in_path = os.path.join("data", "processed", "objects_with_deps.parquet")
    df = pd.read_parquet(in_path)

    results = df["full_path"].apply(estimate_cost)
    df["compression_ratio"] = results.apply(lambda r: r["compression_ratio"] if r else 1.0)
    df["compute_time_sec"] = results.apply(lambda r: r["compute_time_sec"] if r else 0.0)
    df["compressed_size_bytes"] = results.apply(lambda r: r["compressed_size_bytes"] if r else 0)

    # storage cost estimate assuming "hot" tier as the default before any lifecycle action
    df["monthly_storage_cost_usd"] = (df["object_size"] / (1024 ** 3)) * STORAGE_COST_PER_GB_MONTH["hot"]

    out_path = os.path.join("data", "processed", "objects_with_cost.parquet")
    df.to_parquet(out_path)
    print(df[["compression_ratio", "compute_time_sec", "monthly_storage_cost_usd"]].describe())
    print(f"Saved to {out_path}")