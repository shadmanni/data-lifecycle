"""
Walks raw dataset folders, chunks every file, and records:
- file path, source version, size, chunk hash list
Saves memory-optimized shards or compressed parquet file:
data/processed/chunked_objects.parquet (or data/processed/shards/)
"""
import os
import sys
import gc
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.ingestion.chunking import chunk_file
from src.utils.memory_opt import optimize_dtypes, MemoryTracker

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
    """
    Ingests and chunks files for production dataset discovery (used by main.py).
    Memory-optimized with downcasting.
    """
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

    df = optimize_dtypes(pd.DataFrame(rows))
    out_path = os.path.join("data", "processed", "chunked_objects.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)
    print(f"Saved {len(df)} objects to {out_path}", flush=True)
    return df


def walk_and_chunk_stream(
    dataset_versions=None,
    batch_size: int = 5000,
    max_files_per_version: int = 50000,
    out_path: str = "data/processed/chunked_objects.parquet",
    shards_dir: str = "data/processed/shards",
    use_shards: bool = False,
):
    """
    Ingests and chunks raw files using streaming batch writes.
    Maintains O(1) memory overhead by flushing batches to Parquet.
    """
    if dataset_versions is None:
        dataset_versions = get_dataset_versions()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if use_shards:
        os.makedirs(shards_dir, exist_ok=True)

    writer = None
    total_processed = 0
    batch_rows = []
    shard_idx = 0

    with MemoryTracker("Walk and Chunk Files"):
        for version_name, root_dir in dataset_versions.items():
            if not os.path.isdir(root_dir):
                print(f"WARNING: {root_dir} not found, skipping.")
                continue

            count = 0
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in filenames:
                    if count >= max_files_per_version:
                        break
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in VALID_EXTENSIONS:
                        continue

                    full_path = os.path.join(dirpath, fname)
                    try:
                        size = os.path.getsize(full_path)
                        if size == 0 or size > 50_000_000:  # skip empty or huge outlier files (>50MB)
                            continue
                        chunk_hashes = chunk_file(full_path)
                    except Exception:
                        continue

                    batch_rows.append({
                        "object_id": f"{version_name}::{os.path.relpath(full_path, root_dir)}",
                        "version": version_name,
                        "relative_path": os.path.relpath(full_path, root_dir),
                        "full_path": full_path,
                        "object_size": size,
                        "chunk_hashes": chunk_hashes,
                        "num_chunks": len(chunk_hashes),
                    })
                    count += 1
                    total_processed += 1

                    # Flush batch when limit reached
                    if len(batch_rows) >= batch_size:
                        batch_df = optimize_dtypes(pd.DataFrame(batch_rows))
                        table = pa.Table.from_pandas(batch_df, preserve_index=False)

                        if use_shards:
                            shard_file = os.path.join(shards_dir, f"chunked_objects_shard_{shard_idx:04d}.parquet")
                            pq.write_table(table, shard_file, compression="snappy")
                            shard_idx += 1
                        else:
                            if writer is None:
                                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                            writer.write_table(table)

                        batch_rows.clear()
                        gc.collect()

                if count >= max_files_per_version:
                    break

            print(f"Processed {count} files from {version_name}")

        # Flush trailing batch
        if batch_rows:
            batch_df = optimize_dtypes(pd.DataFrame(batch_rows))
            table = pa.Table.from_pandas(batch_df, preserve_index=False)
            if use_shards:
                shard_file = os.path.join(shards_dir, f"chunked_objects_shard_{shard_idx:04d}.parquet")
                pq.write_table(table, shard_file, compression="snappy")
                shard_idx += 1
            else:
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                writer.write_table(table)
            batch_rows.clear()
            gc.collect()

        if writer is not None:
            writer.close()
            print(f"Saved {total_processed:,} objects to {out_path}")
        elif use_shards:
            print(f"Saved {total_processed:,} objects across {shard_idx} shards in {shards_dir}")

    return total_processed


def main():
    parser = argparse.ArgumentParser(description="Streaming File Chunker and Ingestion")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--max_files", type=int, default=50000)
    parser.add_argument("--use_shards", action="store_true")
    args = parser.parse_args()

    walk_and_chunk_stream(
        batch_size=args.batch_size,
        max_files_per_version=args.max_files,
        use_shards=args.use_shards,
    )


if __name__ == "__main__":
    main()