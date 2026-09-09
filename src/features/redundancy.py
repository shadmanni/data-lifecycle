"""
Computes a redundancy_score per object:
fraction of the object's chunks that also appear in >=2 objects across the whole corpus.
"""
import os
import pandas as pd
from collections import Counter


def compute_redundancy_scores(df: pd.DataFrame) -> pd.DataFrame:
    # Build global chunk-hash frequency count across the whole dataset
    chunk_counter = Counter()
    for chunk_list in df["chunk_hashes"]:
        chunk_counter.update(chunk_list)

    def score_row(chunk_list):
        if len(chunk_list) == 0:
            return 0.0
        shared = sum(1 for h in chunk_list if chunk_counter[h] > 1)
        return shared / len(chunk_list)

    df["redundancy_score"] = df["chunk_hashes"].apply(score_row)
    return df


if __name__ == "__main__":
    in_path = os.path.join("data", "processed", "chunked_objects.parquet")
    df = pd.read_parquet(in_path)
    df = compute_redundancy_scores(df)
    out_path = os.path.join("data", "processed", "objects_with_redundancy.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(df[["object_id", "redundancy_score"]].describe())
    print(f"Saved to {out_path}")