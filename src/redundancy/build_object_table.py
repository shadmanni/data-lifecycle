"""
src/redundancy/build_object_table.py
--------------------------------------
Walks the real kernel_versions and gcc_versions folders, chunks every
source file with FastCDC, and computes REAL redundancy features:

  - redundancy_score : fraction of a file's chunks that also appear
                        somewhere else in the same dataset (0-1)
  - exact_duplicate   : 1 if this file's full content is byte-identical
                        to another file anywhere in the dataset
  - version_count     : how many of the N versions contain a file at
                        this same relative path

Output: data/processed/real_objects_raw.csv
    one row per (dataset, version, relative_path) file found.

Run:
    python src/redundancy/build_object_table.py
"""

import os
import hashlib
import csv
from pathlib import Path
from collections import defaultdict
from fastcdc import fastcdc

# ---------------------------------------------------------------
# CONFIG -- adjust these if your paths / limits differ
# ---------------------------------------------------------------
DATASETS = {
    "linux_kernel": {
        "root": Path(r"C:\research\kernel_versions"),
        "versions": ["linux-6.1", "linux-6.2", "linux-6.3", "linux-6.4"],
    },
    "gcc": {
        "root": Path(r"C:\research\gcc_versions"),
        "versions": ["gcc-12.1.0", "gcc-12.2.0", "gcc-12.3.0", "gcc-12.4.0"],
    },
}

# Only chunk text/source-like files -- skip binaries, images, build
# artifacts. Keeps runtime manageable on a CPU-only laptop.
INCLUDE_EXTENSIONS = {
    ".c", ".h", ".cc", ".cpp", ".hpp", ".py", ".s", ".S",
    ".md", ".txt", ".rst", ".def", ".inc", ".y", ".l",
}

# Safety cap so a first run on a laptop finishes in a reasonable time.
# Set to None to process everything (much slower, run overnight).
MAX_FILES_PER_VERSION = 8000

# Skip files bigger than this (bytes) -- huge generated files add
# little research value and slow chunking down a lot.
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB

CHUNK_AVG = 4096
CHUNK_MIN = 1024
CHUNK_MAX = 16384

OUTPUT_PATH = Path("data/processed/real_objects_raw.csv")


def iter_source_files(version_root: Path, max_files=None):
    """Yield (relative_path, full_path, size) for source-like files."""
    count = 0
    for dirpath, _, filenames in os.walk(version_root):
        for fname in filenames:
            ext = Path(fname).suffix
            if ext not in INCLUDE_EXTENSIONS:
                continue
            full_path = Path(dirpath) / fname
            try:
                size = full_path.stat().st_size
            except OSError:
                continue  # broken symlink etc.
            if size == 0 or size > MAX_FILE_SIZE:
                continue
            rel_path = full_path.relative_to(version_root).as_posix()
            yield rel_path, full_path, size
            count += 1
            if max_files is not None and count >= max_files:
                return


def full_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def chunk_hashes(path: Path):
    """Return list of chunk hashes for a file using FastCDC."""
    hashes = []
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return hashes
    if not data:
        return hashes
    for chunk in fastcdc(data, avg_size=CHUNK_AVG, min_size=CHUNK_MIN,
                          max_size=CHUNK_MAX, hf=hashlib.sha256):
        hashes.append(chunk.hash)
    return hashes


def process_dataset(name, root, versions):
    """
    Returns a list of row-dicts for this dataset, with redundancy_score
    and exact_duplicate computed GLOBALLY across all versions of this
    dataset (i.e. a chunk is "shared" if it appears in >1 file, whether
    that's the same relative path in a different version, or a totally
    different file).
    """
    print(f"\n=== Processing dataset: {name} ===")

    # file_id -> metadata
    files_meta = {}
    # chunk_hash -> set of file_ids containing it
    chunk_index = defaultdict(set)
    # full_file_hash -> set of file_ids
    fullhash_index = defaultdict(set)
    # relative_path -> set of versions it appears in
    path_versions = defaultdict(set)

    for version in versions:
        version_root = root / version
        if not version_root.exists():
            print(f"  [WARN] version folder missing: {version_root}")
            continue
        n = 0
        for rel_path, full_path, size in iter_source_files(version_root, MAX_FILES_PER_VERSION):
            file_id = f"{name}::{version}::{rel_path}"
            fhash = full_file_hash(full_path)
            chashes = chunk_hashes(full_path)
            if not chashes:
                continue

            files_meta[file_id] = {
                "dataset": name,
                "version": version,
                "relative_path": rel_path,
                "size_bytes": size,
                "full_hash": fhash,
                "num_chunks": len(chashes),
                "chunk_hashes": chashes,
                "mtime": full_path.stat().st_mtime,
            }
            fullhash_index[fhash].add(file_id)
            path_versions[rel_path].add(version)
            for ch in chashes:
                chunk_index[ch].add(file_id)
            n += 1
        print(f"  {version}: {n} files chunked")

    print(f"  Total files in dataset: {len(files_meta)}")
    print(f"  Total unique chunks seen: {len(chunk_index)}")

    # Now compute redundancy_score + exact_duplicate per file
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
    all_rows = []
    for name, cfg in DATASETS.items():
        rows = process_dataset(name, cfg["root"], cfg["versions"])
        all_rows.extend(rows)

    if not all_rows:
        print("No rows produced -- check your dataset paths in the CONFIG section.")
        return

    fieldnames = list(all_rows[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to {OUTPUT_PATH}")
    avg_redundancy = sum(r["redundancy_score"] for r in all_rows) / len(all_rows)
    exact_dupes = sum(r["exact_duplicate"] for r in all_rows)
    print(f"Average redundancy_score across all files: {avg_redundancy:.3f}")
    print(f"Files flagged as exact duplicates: {exact_dupes} / {len(all_rows)}")


if __name__ == "__main__":
    main()