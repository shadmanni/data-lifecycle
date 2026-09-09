"""
verify_data.py
---------------
Phase 1.3: Verify the 3 raw datasets downloaded so far are complete
and not corrupted, before building any pipeline on top of them.

Checks per dataset:
  - Does the expected top-level folder/files exist?
  - How many files total, and total size?
  - A couple of spot-check files that MUST exist if extraction worked.

Run this from inside lifecycle-project with the venv activated:
    python src/ingestion/verify_data.py
"""

import os
from pathlib import Path

# Adjust these if your paths differ
KERNEL_DIR = Path(r"C:\research\kernel_versions")
GCC_DIR = Path(r"C:\research\gcc_versions")
AZURE_DIR = Path(r"C:\research\azure_functions_2019")

KERNEL_VERSIONS = ["linux-6.1", "linux-6.2", "linux-6.3", "linux-6.4"]
GCC_VERSIONS = ["gcc-12.1.0", "gcc-12.2.0", "gcc-12.3.0", "gcc-12.4.0"]


def folder_stats(path: Path):
    """Count files and total size (bytes) under a folder, recursively."""
    if not path.exists():
        return None
    file_count = 0
    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total_size += fp.stat().st_size
                file_count += 1
            except OSError:
                # broken symlinks etc. -- count but don't crash
                file_count += 1
    return file_count, total_size


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def check_kernel():
    print("=" * 70)
    print("LINUX KERNEL VERSIONS")
    print("=" * 70)
    all_ok = True
    for v in KERNEL_VERSIONS:
        vpath = KERNEL_DIR / v
        stats = folder_stats(vpath)
        if stats is None:
            print(f"  [MISSING] {v} -- folder not found at {vpath}")
            all_ok = False
            continue
        count, size = stats
        # sanity check: a real kernel extraction should have well over
        # 50,000 files and be at least a few hundred MB
        ok = count > 50000 and size > 300 * 1024 * 1024
        flag = "OK" if ok else "SUSPICIOUS"
        print(f"  [{flag}] {v}: {count:,} files, {human_size(size)}")
        # spot-check a file we know should exist
        spot = vpath / "kernel" / "sched" / "core.c"
        print(f"         spot-check kernel/sched/core.c exists: {spot.exists()}")
        if not ok:
            all_ok = False
    return all_ok


def check_gcc():
    print("=" * 70)
    print("GCC VERSIONS")
    print("=" * 70)
    all_ok = True
    for v in GCC_VERSIONS:
        vpath = GCC_DIR / v
        stats = folder_stats(vpath)
        if stats is None:
            print(f"  [MISSING] {v} -- folder not found at {vpath}")
            all_ok = False
            continue
        count, size = stats
        # GCC source trees are large -- expect tens of thousands of files, GBs
        ok = count > 30000 and size > 500 * 1024 * 1024
        flag = "OK" if ok else "SUSPICIOUS"
        print(f"  [{flag}] {v}: {count:,} files, {human_size(size)}")
        spot = vpath / "gcc" / "gcc.cc"
        print(f"         spot-check gcc/gcc.cc exists: {spot.exists()}")
        if not ok:
            all_ok = False
    return all_ok


def check_azure():
    print("=" * 70)
    print("AZURE FUNCTIONS 2019")
    print("=" * 70)
    if not AZURE_DIR.exists():
        print(f"  [MISSING] folder not found at {AZURE_DIR}")
        return False

    invocation_files = sorted(AZURE_DIR.glob("invocations_per_function_md.anon.d*.csv"))
    duration_files = sorted(AZURE_DIR.glob("function_durations_percentiles.anon.d*.csv"))
    memory_files = sorted(AZURE_DIR.glob("app_memory_percentiles.anon.d*.csv"))

    print(f"  Invocation-count files found: {len(invocation_files)} (expected 14)")
    print(f"  Duration files found:         {len(duration_files)} (expected 14)")
    print(f"  Memory files found:           {len(memory_files)} (expected 12)")

    all_ok = len(invocation_files) > 0  # at minimum we need the invocation files

    if invocation_files:
        first = invocation_files[0]
        size = first.stat().st_size
        print(f"  Sample file: {first.name}, size = {human_size(size)}")
        ok = size > 1 * 1024 * 1024  # should be several MB at least
        if not ok:
            print("  [SUSPICIOUS] sample file seems too small")
            all_ok = False
        else:
            print("  [OK] sample file size looks reasonable")

        # peek at the header to confirm expected columns
        with open(first, "r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().strip()
        expected_cols = ["HashOwner", "HashApp", "HashFunction", "Trigger"]
        has_expected = all(c in header for c in expected_cols)
        print(f"  Header contains expected columns {expected_cols}: {has_expected}")
        if not has_expected:
            all_ok = False
    else:
        all_ok = False

    return all_ok


def main():
    results = {
        "Linux Kernel": check_kernel(),
        "GCC": check_gcc(),
        "Azure Functions 2019": check_azure(),
    }

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL / NEEDS ATTENTION'}")

    if all(results.values()):
        print("\nAll datasets verified. Safe to proceed to Phase 2 (chunking / feature engineering).")
    else:
        print("\nSome datasets failed verification -- check paths and re-extract if needed before proceeding.")


if __name__ == "__main__":
    main()