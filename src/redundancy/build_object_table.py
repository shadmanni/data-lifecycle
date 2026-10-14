"""
src/redundancy/build_object_table.py
--------------------------------------
Walks the NASA Turbofan Jet Engine (CMaps) dataset:
- Primary input path: /Users/shifashameer/Downloads/CMaps
- Subsets / versions: FD001, FD002, FD003, FD004
- Safely ignores non-data documentation (readme.txt, Damage Propagation Modeling.pdf)
- Chunks every file / telemetry partition with FastCDC / Gear chunking
- Computes real redundancy features:
    * redundancy_score: fraction of chunks that appear in >=2 objects
    * exact_duplicate: 1 if full content is byte-identical to another object
    * version_count: how many versions contain this object type
Output: data/processed/real_objects_raw.csv
"""

import os
import sys
import hashlib
import csv
from pathlib import Path
from collections import defaultdict

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingestion.chunking import chunk_file
from src.ingestion.cmaps_loader import (
    DEFAULT_CMAPS_DIR,
    resolve_cmaps_dir,
    is_valid_cmaps_data_file,
    is_ignored_cmaps_file,
    DATA_FILE_PATTERN,
)

OUTPUT_PATH = Path("data/processed/real_objects_raw.csv")


def full_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def chunk_hashes(path: Path):
    """Return list of chunk hashes for a file using pure-Python Gear/FastCDC chunker."""
    return chunk_file(str(path))


def process_cmaps(root_path: Path):
    """
    Process CMaps files from root directory:
    - Filters only valid data files (train, test, RUL)
    - Safely ignores readme.txt, Damage Propagation Modeling.pdf, etc.
    - Computes global chunk index, duplicate index, and version sharing
    """
    print(f"\n=== Processing CMaps Dataset at: {root_path} ===")
    if not root_path.exists():
        print(f"  [WARN] CMaps directory not found at: {root_path}")
        return []

    files_meta = {}
    chunk_index = defaultdict(set)
    fullhash_index = defaultdict(set)
    path_versions = defaultdict(set)

    ignored_files = []
    data_files = []

    for entry in sorted(os.listdir(root_path)):
        full_p = root_path / entry
        if not full_p.is_file():
            continue

        if is_ignored_cmaps_file(entry):
            ignored_files.append(entry)
            continue

        match = DATA_FILE_PATTERN.match(entry)
        if match:
            kind, subset = match.group(1).lower(), match.group(2).upper()
            data_files.append((entry, full_p, kind, subset))

    if ignored_files:
        print(f"  Safely ignored {len(ignored_files)} non-data documentation file(s): {ignored_files}")

    for entry, full_p, kind, subset in data_files:
        file_id = f"CMaps::{subset}::{entry}"
        size = full_p.stat().st_size
        if size == 0:
            continue

        fhash = full_file_hash(full_p)
        chashes = chunk_hashes(full_p)
        if not chashes:
            continue

        files_meta[file_id] = {
            "dataset": "CMaps",
            "version": subset,
            "relative_path": entry,
            "size_bytes": size,
            "full_hash": fhash,
            "num_chunks": len(chashes),
            "chunk_hashes": chashes,
            "mtime": full_p.stat().st_mtime,
        }
        fullhash_index[fhash].add(file_id)
        path_versions[entry].add(subset)
        for ch in chashes:
            chunk_index[ch].add(file_id)

    # Also include partitioned engine units if present in data/processed/cmaps_partitions
    partitions_dir = Path("data/processed/cmaps_partitions")
    if partitions_dir.is_dir():
        for pfile in sorted(partitions_dir.glob("*.txt")):
            fname = pfile.name
            subset = fname.split("_")[0] if "_" in fname else "FD001"
            file_id = f"CMaps::{subset}::partition::{fname}"
            size = pfile.stat().st_size
            if size == 0:
                continue
            fhash = full_file_hash(pfile)
            chashes = chunk_hashes(pfile)
            if not chashes:
                continue

            files_meta[file_id] = {
                "dataset": "CMaps",
                "version": subset,
                "relative_path": fname,
                "size_bytes": size,
                "full_hash": fhash,
                "num_chunks": len(chashes),
                "chunk_hashes": chashes,
                "mtime": pfile.stat().st_mtime,
            }
            fullhash_index[fhash].add(file_id)
            path_versions[fname].add(subset)
            for ch in chashes:
                chunk_index[ch].add(file_id)

    print(f"  Total CMaps objects processed: {len(files_meta)}")
    print(f"  Total unique chunks indexed: {len(chunk_index)}")

    rows = []
    for file_id, meta in files_meta.items():
        chashes = meta["chunk_hashes"]
        shared_count = sum(1 for ch in chashes if len(chunk_index[ch]) > 1)
        redundancy_score = shared_count / len(chashes) if chashes else 0.0
        exact_duplicate = 1 if len(fullhash_index[meta["full_hash"]]) > 1 else 0
        version_count = len(path_versions[meta["relative_path"]])

        rows.append({
            "dataset": meta["dataset"],
            "version": meta["version"],
            "relative_path": meta["relative_path"],
            "size_bytes": meta["size_bytes"],
            "num_chunks": meta["num_chunks"],
            "redundancy_score": round(redundancy_score, 4),
            "exact_duplicate": exact_duplicate,
            "version_count": version_count,
            "mtime": meta["mtime"],
        })

    return rows


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_dir = Path(resolve_cmaps_dir())

    all_rows = process_cmaps(target_dir)

    if not all_rows:
        print(f"No objects produced from {target_dir}. Check directory contents.")
        return

    fieldnames = list(all_rows[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to {OUTPUT_PATH}")
    avg_redundancy = sum(r["redundancy_score"] for r in all_rows) / len(all_rows)
    exact_dupes = sum(r["exact_duplicate"] for r in all_rows)
    print(f"Average redundancy_score: {avg_redundancy:.4f}")
    print(f"Exact duplicates detected: {exact_dupes} / {len(all_rows)}")


if __name__ == "__main__":
    main()