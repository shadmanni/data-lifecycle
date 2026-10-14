"""
build_objects.py
-----------------
Walks the NASA Turbofan Jet Engine (CMaps) dataset:
- Primary input path: /Users/shifashameer/Downloads/CMaps
- Handles file structure: train_FD001-4.txt, test_FD001-4.txt, RUL_FD001-4.txt
- Safely ignores readme.txt, Damage Propagation Modeling.pdf, etc.
- Parses space-separated text, assigns column names, supplies missing RUL index
- Partitions engine telemetry runs and master dataset files into cloud data objects
- Chunks every object with content-defined chunking (Gear / FastCDC)
- Saves intermediate parquet: data/processed/chunked_objects.parquet
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingestion.chunking import chunk_file
from src.ingestion.cmaps_loader import (
    DEFAULT_CMAPS_DIR,
    resolve_cmaps_dir,
    parse_cmaps_file,
    parse_rul_file,
    is_valid_cmaps_data_file,
    is_ignored_cmaps_file,
    CMAPS_COLUMNS,
    DATA_FILE_PATTERN,
)

MAX_ENGINES_PER_FILE = 100  # Cap per file for responsive, balanced multi-version chunking


def get_dataset_versions(cmaps_dir: str = None) -> dict:
    """
    Dynamically discover CMaps dataset versions (FD001, FD002, FD003, FD004)
    from the local path (default: /Users/shifashameer/Downloads/CMaps).
    """
    resolved_dir = resolve_cmaps_dir(cmaps_dir)
    versions = {}

    if os.path.isdir(resolved_dir):
        # Scan for CMaps subsets FD001, FD002, FD003, FD004
        for entry in sorted(os.listdir(resolved_dir)):
            full_path = os.path.join(resolved_dir, entry)
            if not os.path.isfile(full_path):
                continue
            if is_ignored_cmaps_file(entry):
                continue

            match = DATA_FILE_PATTERN.match(entry)
            if match:
                subset = match.group(2).upper()
                if subset not in versions:
                    versions[subset] = {}
                kind = match.group(1).lower()
                versions[subset][kind] = full_path

    # Fallback to legacy versions if CMaps not yet populated
    if not versions:
        legacy_raw = os.path.join("data", "raw")
        kernel_dir = os.path.join(legacy_raw, "kernel_versions")
        if os.path.isdir(kernel_dir):
            for entry in sorted(os.listdir(kernel_dir)):
                p = os.path.join(kernel_dir, entry)
                if os.path.isdir(p):
                    versions[entry] = p
        gcc_dir = os.path.join(legacy_raw, "gcc_versions")
        if os.path.isdir(gcc_dir):
            for entry in sorted(os.listdir(gcc_dir)):
                p = os.path.join(gcc_dir, entry)
                if os.path.isdir(p):
                    versions[entry] = p

    return versions


def partition_cmaps_objects(
    cmaps_dir: str,
    output_partitions_dir: str = os.path.join("data", "processed", "cmaps_partitions"),
    max_engines: int = MAX_ENGINES_PER_FILE,
) -> list:
    """
    Parse space-separated CMaps files, safely ignoring non-data files (readme.txt, pdf).
    Partitions each engine's telemetry into individual object files for cloud lifecycle tracking.
    Returns metadata dicts for all objects (both master files and partitioned engine objects).
    """
    os.makedirs(output_partitions_dir, exist_ok=True)
    all_objects = []

    entries = sorted(os.listdir(cmaps_dir))
    ignored_count = 0

    for entry in entries:
        full_path = os.path.join(cmaps_dir, entry)
        if not os.path.isfile(full_path):
            continue

        if is_ignored_cmaps_file(entry):
            ignored_count += 1
            continue

        match = DATA_FILE_PATTERN.match(entry)
        if not match:
            continue

        kind = match.group(1).lower()  # train, test, rul
        subset = match.group(2).upper()  # FD001, etc.
        file_size = os.path.getsize(full_path)
        if file_size == 0:
            continue

        # 1. Master File Object
        file_hashes = chunk_file(full_path)
        all_objects.append({
            "object_id": f"{subset}::{entry}",
            "version": subset,
            "relative_path": entry,
            "full_path": full_path,
            "object_size": file_size,
            "chunk_hashes": file_hashes,
            "num_chunks": len(file_hashes),
        })

        # 2. Parse Space-Separated Content & Partition Engines into Telemetry Objects
        try:
            if kind in ("train", "test"):
                df = parse_cmaps_file(full_path)
                units = df["unit_number"].unique()[:max_engines]
                for unit_id in units:
                    unit_df = df[df["unit_number"] == unit_id]
                    part_fname = f"{subset}_{kind}_unit_{unit_id:03d}.txt"
                    part_path = os.path.join(output_partitions_dir, part_fname)

                    # Save formatted space-separated engine telemetry object
                    unit_df.to_csv(part_path, sep=" ", index=False, header=False)
                    part_size = os.path.getsize(part_path)
                    part_hashes = chunk_file(part_path)

                    all_objects.append({
                        "object_id": f"{subset}::{kind}::unit_{unit_id:03d}",
                        "version": subset,
                        "relative_path": f"{subset}/{kind}/unit_{unit_id:03d}.txt",
                        "full_path": part_path,
                        "object_size": part_size,
                        "chunk_hashes": part_hashes,
                        "num_chunks": len(part_hashes),
                    })
            elif kind == "rul":
                rul_df = parse_rul_file(full_path)
                part_fname = f"{subset}_rul_groundtruth.csv"
                part_path = os.path.join(output_partitions_dir, part_fname)
                rul_df.to_csv(part_path, index=False)
                part_hashes = chunk_file(part_path)
                all_objects.append({
                    "object_id": f"{subset}::rul::groundtruth",
                    "version": subset,
                    "relative_path": f"{subset}/rul/groundtruth.csv",
                    "full_path": part_path,
                    "object_size": os.path.getsize(part_path),
                    "chunk_hashes": part_hashes,
                    "num_chunks": len(part_hashes),
                })
        except Exception as e:
            print(f"Notice: Failed to partition {entry}: {e}", flush=True)

    if ignored_count > 0:
        print(f"Safely ignored {ignored_count} non-data documentation files (e.g. readme.txt, pdf).", flush=True)

    return all_objects


def walk_and_chunk(discovered_versions: dict = None, cmaps_dir: str = None) -> pd.DataFrame:
    """
    Main ingestion & FastCDC chunking function.
    Reads CMaps data from resolved path (default /Users/shifashameer/Downloads/CMaps),
    handles parsing & missing indices, chunks telemetry, and exports chunked_objects.parquet.
    """
    resolved_dir = resolve_cmaps_dir(cmaps_dir)
    print(f"Ingesting CMaps dataset from: {resolved_dir}", flush=True)

    rows = []

    if os.path.isdir(resolved_dir):
        # CMaps dataset ingestion
        rows = partition_cmaps_objects(resolved_dir)

    # Fallback to general walk if no CMaps rows found
    if not rows and discovered_versions:
        for vname, vval in discovered_versions.items():
            root_dir = vval if isinstance(vval, str) else vval.get("train", "")
            if not os.path.isdir(root_dir):
                continue
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in filenames:
                    if is_ignored_cmaps_file(fname):
                        continue
                    full_p = os.path.join(dirpath, fname)
                    try:
                        sz = os.path.getsize(full_p)
                        if sz == 0:
                            continue
                        hashes = chunk_file(full_p)
                        rows.append({
                            "object_id": f"{vname}::{os.path.relpath(full_p, root_dir)}",
                            "version": vname,
                            "relative_path": os.path.relpath(full_p, root_dir),
                            "full_path": full_p,
                            "object_size": sz,
                            "chunk_hashes": hashes,
                            "num_chunks": len(hashes),
                        })
                    except Exception:
                        continue

    df = pd.DataFrame(rows)
    out_path = os.path.join("data", "processed", "chunked_objects.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"Saved {len(df)} objects across {df['version'].nunique() if len(df) else 0} versions to {out_path}", flush=True)
    return df


if __name__ == "__main__":
    discovered = get_dataset_versions()
    df = walk_and_chunk(discovered)