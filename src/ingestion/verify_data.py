"""
verify_data.py
---------------
Verifies the NASA Turbofan Jet Engine (CMaps) dataset:
- Direct primary path: /Users/shifashameer/Downloads/CMaps
- Verifies all 4 training files: train_FD001.txt to train_FD004.txt
- Verifies all 4 test files: test_FD001.txt to test_FD004.txt
- Verifies all 4 RUL files: RUL_FD001.txt to RUL_FD004.txt
- Confirms non-data files (readme.txt, Damage Propagation Modeling.pdf) are safely ignored
- Verifies space-separated parsing, column structure (26 cols for train/test), and missing index handling for RUL
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingestion.cmaps_loader import (
    DEFAULT_CMAPS_DIR,
    resolve_cmaps_dir,
    parse_cmaps_file,
    parse_rul_file,
    is_valid_cmaps_data_file,
    is_ignored_cmaps_file,
    CMAPS_COLUMNS,
)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def check_cmaps_dataset(cmaps_dir: str = DEFAULT_CMAPS_DIR) -> bool:
    print("=" * 75)
    print("NASA TURBOFAN JET ENGINE (CMAPS) DATASET VERIFICATION")
    print(f"Target Directory: {cmaps_dir}")
    print("=" * 75)

    dir_path = Path(cmaps_dir)
    if not dir_path.exists():
        print(f"  [MISSING] Target directory not found at: {cmaps_dir}")
        print("  Notice: When running on Shifa's Mac, ensure the dataset is in /Users/shifashameer/Downloads/CMaps.")
        return False

    all_ok = True
    found_files = set(os.listdir(dir_path))

    # 1. Check Non-Data / Ignored Files
    doc_files = [f for f in found_files if is_ignored_cmaps_file(f)]
    print(f"\n[1] Documentation & Ignored Files Check:")
    print(f"  Detected non-data files: {doc_files}")
    for doc in doc_files:
        print(f"  [SAFELY IGNORED] {doc} -> correctly bypassed by data loader.")

    # 2. Check Training Files
    print(f"\n[2] Training Files (train_FD001.txt - train_FD004.txt):")
    for sub in SUBSETS:
        fname = f"train_{sub}.txt"
        fpath = dir_path / fname
        if not fpath.exists():
            print(f"  [MISSING] {fname}")
            all_ok = False
            continue

        size = fpath.stat().st_size
        try:
            df = parse_cmaps_file(str(fpath))
            n_rows, n_cols = df.shape
            units = df["unit_number"].nunique()
            print(
                f"  [OK] {fname:<15}: {human_size(size):>8} | {n_rows:>6} rows | "
                f"{n_cols} cols | {units:>3} engines"
            )
            # Verify 26 columns and correct headers
            if n_cols != 26 or list(df.columns) != CMAPS_COLUMNS:
                print(f"       [ERROR] Header mismatch or column count != 26 in {fname}")
                all_ok = False
        except Exception as e:
            print(f"  [ERROR] Failed to parse {fname}: {e}")
            all_ok = False

    # 3. Check Test Files
    print(f"\n[3] Test Files (test_FD001.txt - test_FD004.txt):")
    for sub in SUBSETS:
        fname = f"test_{sub}.txt"
        fpath = dir_path / fname
        if not fpath.exists():
            print(f"  [MISSING] {fname}")
            all_ok = False
            continue

        size = fpath.stat().st_size
        try:
            df = parse_cmaps_file(str(fpath))
            n_rows, n_cols = df.shape
            units = df["unit_number"].nunique()
            print(
                f"  [OK] {fname:<15}: {human_size(size):>8} | {n_rows:>6} rows | "
                f"{n_cols} cols | {units:>3} engines"
            )
            if n_cols != 26 or list(df.columns) != CMAPS_COLUMNS:
                print(f"       [ERROR] Header mismatch or column count != 26 in {fname}")
                all_ok = False
        except Exception as e:
            print(f"  [ERROR] Failed to parse {fname}: {e}")
            all_ok = False

    # 4. Check RUL Files (Handling missing index column)
    print(f"\n[4] RUL Files (RUL_FD001.txt - RUL_FD004.txt):")
    for sub in SUBSETS:
        fname = f"RUL_{sub}.txt"
        fpath = dir_path / fname
        if not fpath.exists():
            print(f"  [MISSING] {fname}")
            all_ok = False
            continue

        size = fpath.stat().st_size
        try:
            rul_df = parse_rul_file(str(fpath))
            n_rows, n_cols = rul_df.shape
            has_unit_idx = "unit_number" in rul_df.columns
            print(
                f"  [OK] {fname:<15}: {human_size(size):>8} | {n_rows:>6} units | "
                f"missing unit_number index supplied: {has_unit_idx}"
            )
            if not has_unit_idx or n_cols != 2:
                print(f"       [ERROR] Missing index column resolution failed for {fname}")
                all_ok = False
        except Exception as e:
            print(f"  [ERROR] Failed to parse {fname}: {e}")
            all_ok = False

    return all_ok


def main():
    target_dir = resolve_cmaps_dir()
    ok = check_cmaps_dataset(target_dir)

    print("\n" + "=" * 75)
    print("VERIFICATION SUMMARY")
    print("=" * 75)
    print(f"  CMaps Dataset at '{target_dir}': {'PASS' if ok else 'FAIL / ACTION NEEDED'}")

    if ok:
        print("\nAll CMaps files verified and parsed successfully.")
    else:
        print("\nPlease ensure all CMaps files are downloaded to the target directory.")


if __name__ == "__main__":
    main()