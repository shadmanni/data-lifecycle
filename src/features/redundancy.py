"""
Computes a redundancy_score per object:
fraction of the object's chunks that also appear in >=2 objects across the whole corpus.
Memory-optimized with dtype downcasting and streaming support.
"""
import os
import sys
import glob
import pandas as pd
from collections import Counter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.memory_opt import optimize_dtypes, MemoryTracker


def compute_redundancy_scores(df: pd.DataFrame, drop_chunk_hashes: bool = True) -> pd.DataFrame:
    """
    Computes redundancy score and downcasts result to float32.
    Optionally drops heavy chunk_hashes list to conserve RAM.
    """
    with MemoryTracker("Compute Redundancy Scores"):
        chunk_counter = Counter()
        for chunk_list in df["chunk_hashes"]:
            chunk_counter.update(chunk_list)

        def score_row(chunk_list):
            if chunk_list is None or len(chunk_list) == 0:
                return 0.0
            shared = sum(1 for h in chunk_list if chunk_counter[h] > 1)
            return float(shared / len(chunk_list))

        df["redundancy_score"] = df["chunk_hashes"].apply(score_row).astype("float32")

        if drop_chunk_hashes and "chunk_hashes" in df.columns:
            df = df.drop(columns=["chunk_hashes"])

        df = optimize_dtypes(df, verbose=True)
    return df


def process_dataset(
    in_path: str = "data/processed/chunked_objects.parquet",
    shards_dir: str = "data/processed/shards",
    out_path: str = "data/processed/objects_with_redundancy.parquet",
    drop_chunk_hashes: bool = True,
):
    """
    Reads either consolidated chunked_objects.parquet or loads shards iteratively.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    shard_files = sorted(glob.glob(os.path.join(shards_dir, "*.parquet"))) if shards_dir and os.path.isdir(shards_dir) else []
    if shard_files:
        print(f"Loading {len(shard_files)} shards from {shards_dir}...")
        df = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
    elif in_path and os.path.exists(in_path):
        print(f"Reading consolidated objects from {in_path}...")
        df = pd.read_parquet(in_path)
    else:
        raise FileNotFoundError(f"Neither shards in '{shards_dir}' nor file '{in_path}' found!")

    df = compute_redundancy_scores(df, drop_chunk_hashes=drop_chunk_hashes)
    df.to_parquet(out_path, compression="snappy", index=False)
    print(f"Saved {len(df):,} objects with redundancy scores to {out_path}")
    return df


if __name__ == "__main__":
    process_dataset()
