"""
Estimates compression benefit and compute cost per object using zlib
(built into Python, works identically on every OS -- avoids xdelta/Windows issues).
Includes fallback proxy estimation for large-scale synthetic datasets.
Memory-optimized with downcasting to float32 and uint32.
"""
import os
import sys
import time
import zlib
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.memory_opt import optimize_dtypes, MemoryTracker

STORAGE_COST_PER_GB_MONTH = {
    "hot": 0.023,
    "cool": 0.0125,
    "cold": 0.004,
    "archive": 0.00099,
}


def estimate_cost(filepath: str, fallback_size: int = 100000):
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            start = time.perf_counter()
            compressed = zlib.compress(data, level=6)
            elapsed = time.perf_counter() - start
            original_size = len(data)
            compressed_size = len(compressed)
            ratio = original_size / max(1, compressed_size)
            return {
                "compression_ratio": float(ratio),
                "compute_time_sec": float(elapsed),
                "compressed_size_bytes": compressed_size,
            }
        except Exception:
            pass

    # Fallback simulation for synthetic shards without local raw files
    ratio = float(np.random.uniform(1.8, 3.5))
    compressed_size = int(fallback_size / ratio)
    elapsed = float(fallback_size / (50 * 1024 * 1024))  # ~50MB/sec compression rate proxy
    return {
        "compression_ratio": ratio,
        "compute_time_sec": elapsed,
        "compressed_size_bytes": compressed_size,
    }


def process_costs(
    in_path: str = "data/processed/objects_with_deps.parquet",
    out_path: str = "data/processed/objects_with_cost.parquet",
):
    with MemoryTracker("Cost and Compression Estimation"):
        df = pd.read_parquet(in_path)

        first_path = df["full_path"].iloc[0] if not df.empty else ""
        if os.path.exists(first_path):
            compression_ratios = []
            compute_times = []
            compressed_sizes = []
            for _, row in df.iterrows():
                fpath = row.get("full_path", "")
                fsize = int(row.get("object_size", 100000))
                res = estimate_cost(fpath, fallback_size=fsize)
                compression_ratios.append(res["compression_ratio"])
                compute_times.append(res["compute_time_sec"])
                compressed_sizes.append(res["compressed_size_bytes"])
            df["compression_ratio"] = np.array(compression_ratios, dtype=np.float32)
            df["compute_time_sec"] = np.array(compute_times, dtype=np.float32)
            df["compressed_size_bytes"] = np.array(compressed_sizes, dtype=np.uint32)
        else:
            # High-performance vectorized synthetic estimation
            n = len(df)
            ratios = np.random.uniform(1.8, 3.5, size=n).astype(np.float32)
            sizes = df["object_size"].to_numpy(dtype=np.float32)
            df["compression_ratio"] = ratios
            df["compressed_size_bytes"] = (sizes / ratios).astype(np.uint32)
            df["compute_time_sec"] = (sizes / (50.0 * 1024 * 1024)).astype(np.float32)

        # storage cost estimate assuming "hot" tier as default
        df["monthly_storage_cost_usd"] = (
            (df["object_size"].astype(np.float32) / (1024 ** 3)) * STORAGE_COST_PER_GB_MONTH["hot"]
        ).astype(np.float32)

        df = optimize_dtypes(df, verbose=True)
        df.to_parquet(out_path, compression="snappy", index=False)
        print(df[["compression_ratio", "compute_time_sec", "monthly_storage_cost_usd"]].describe())
        print(f"Saved {len(df):,} objects with cost estimates to {out_path}")
    return df


if __name__ == "__main__":
    process_costs()