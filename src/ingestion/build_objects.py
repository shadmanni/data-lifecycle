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

# Process source code and trace data extensions
VALID_EXTENSIONS = {".c", ".h", ".cpp", ".hpp", ".py", ".md", ".txt", ".cc", ".csv"}

RAW_DATA_DIR = os.path.join("data", "raw")


def get_dataset_versions(raw_dir=RAW_DATA_DIR):
    """Dynamically discover production dataset versions or fall back to known paths."""
    versions = {}
    
    # Kernel versions
    kernel_dir = os.path.join(raw_dir, "kernel_versions")
    if os.path.isdir(kernel_dir):
        for entry in sorted(os.listdir(kernel_dir)):
            full_path = os.path.join(kernel_dir, entry)
            if os.path.isdir(full_path):
                versions[entry] = full_path

    # GCC versions
    gcc_dir = os.path.join(raw_dir, "gcc_versions")
    if os.path.isdir(gcc_dir):
        for entry in sorted(os.listdir(gcc_dir)):
            full_path = os.path.join(gcc_dir, entry)
            if os.path.isdir(full_path):
                versions[entry] = full_path

    # Azure Functions 2019 traces
    azure_dir = os.path.join(raw_dir, "azure_functions_2019")
    if os.path.isdir(azure_dir):
        versions["azure_functions_2019"] = azure_dir

    # Fallback to synthetic if raw is missing
    if not versions:
        versions = {
            "synth_v1": os.path.join("data", "raw", "synth_v1"),
            "synth_v2": os.path.join("data", "raw", "synth_v2"),
            "synth_v3": os.path.join("data", "raw", "synth_v3"),
        }
    return versions


DATASET_VERSIONS = get_dataset_versions()
MAX_FILES_PER_VERSION = 250  # Cap per version for fast, responsive multi-version chunking


def walk_and_chunk(dataset_versions=None, max_files=MAX_FILES_PER_VERSION):
    if dataset_versions is None:
        dataset_versions = get_dataset_versions()

    rows = []
    for version_name, root_dir in dataset_versions.items():
        if not os.path.isdir(root_dir):
            print(f"WARNING: {root_dir} not found, skipping.", flush=True)
            continue

        count = 0
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if count >= max_files:
                    break
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue

                full_path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(full_path)
                    if size == 0:
                        continue
                    chunk_hashes = chunk_file(full_path)
                except Exception:
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
            if count >= max_files:
                break

        print(f"  Processed {count} files from {version_name}", flush=True)

    df = pd.DataFrame(rows)
    out_path = os.path.join("data", "processed", "chunked_objects.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"Saved {len(df)} objects to {out_path}", flush=True)
    return df


if __name__ == "__main__":
    walk_and_chunk()