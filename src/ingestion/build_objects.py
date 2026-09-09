"""
Walks the raw dataset folders, chunks every file, and records:
- file path, source version, size, chunk hash list
Saves an intermediate parquet file: data/processed/chunked_objects.parquet
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
from chunking import chunk_file

# Only process these extensions to keep runtime reasonable for a first pass
VALID_EXTENSIONS = {".c", ".h", ".cpp", ".hpp", ".py", ".md", ".txt", ".cc"}

# Each dataset "version group" -> used later for redundancy comparison and time-based splitting
DATASET_VERSIONS = {
    "synth_v1": os.path.join("data", "raw", "synth_v1"),
    "synth_v2": os.path.join("data", "raw", "synth_v2"),
    "synth_v3": os.path.join("data", "raw", "synth_v3"),
}

MAX_FILES_PER_VERSION = 3000  # cap for a manageable first run; raise later once pipeline works


def walk_and_chunk():
    rows = []
    for version_name, root_dir in DATASET_VERSIONS.items():
        if not os.path.isdir(root_dir):
            print(f"WARNING: {root_dir} not found, skipping. Did Phase 1 downloads finish?")
            continue

        count = 0
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if count >= MAX_FILES_PER_VERSION:
                    break
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue

                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                    if size == 0 or size > 5_000_000:  # skip empty or huge outlier files
                        continue
                    chunk_hashes = chunk_file(full_path)
                except Exception as e:
                    continue

                rows.append({
                    "object_id": f"{version_name}::{os.path.relpath(full_path, root_dir)}",
                    "version": version_name,
                    "relative_path": os.path.relpath(full_path, root_dir),
                    "full_path": full_path,
                    "object_size": size,
                    "chunk_hashes": chunk_hashes,
                    "num_chunks": len(chunk_hashes),
                })
                count += 1
            if count >= MAX_FILES_PER_VERSION:
                break

        print(f"Processed {count} files from {version_name}")

    df = pd.DataFrame(rows)
    out_path = os.path.join("data", "processed", "chunked_objects.parquet")
    df.to_parquet(out_path)
    print(f"Saved {len(df)} objects to {out_path}")
    return df


if __name__ == "__main__":
    walk_and_chunk()