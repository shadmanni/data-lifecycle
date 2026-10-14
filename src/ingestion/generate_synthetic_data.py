"""
Scalable Synthetic Data Generator for Data Lifecycle Management.
Supports programmatically scaling from megabytes to 10GB+ across customizable tiers.
Uses streaming batch generation and compressed disk shards (.parquet) to avoid OOM on laptop RAM.
"""
import os
import sys
import argparse
import random
import gc
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.ingestion.chunking import gear_chunks, hash_chunk
from src.utils.memory_opt import optimize_dtypes, MemoryTracker, log_memory

WORDS = [
    "int", "void", "return", "struct", "char", "float", "if", "else", "for",
    "while", "static", "const", "define", "include", "typedef", "malloc",
    "free", "printf", "buffer", "size", "index", "value", "config", "handler",
    "cache", "queue", "mutex", "socket", "packet", "status", "offset", "length",
    "record", "header", "payload", "stream", "filter", "pipeline", "segment"
]

SIZE_TIERS = {
    "mini": 0.05,       # ~50 MB
    "small": 2.0,       # ~2 GB baseline
    "medium": 5.0,      # ~5 GB
    "large": 10.0,      # ~10 GB target
    "xlarge": 20.0,     # ~20 GB
}


def make_function_block_bytes(rng: random.Random, num_lines: int = 15) -> bytes:
    """Generates a synthetic function block as UTF-8 bytes."""
    lines = []
    for _ in range(num_lines):
        line = " ".join(rng.choices(WORDS, k=rng.randint(4, 10))) + ";"
        lines.append(line)
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def generate_object_payload(rng: random.Random, num_blocks: int, mutation_ratio: float = 0.0) -> bytes:
    """
    Generates synthetic file bytes. If mutation_ratio > 0, randomly alters
    certain blocks to simulate realistic version-to-version edits.
    """
    blocks = []
    for _ in range(num_blocks):
        if mutation_ratio > 0 and rng.random() < mutation_ratio:
            # Modified block
            mutated_rng = random.Random(rng.randint(0, 1000000))
            blocks.append(make_function_block_bytes(mutated_rng, num_lines=rng.randint(8, 20)))
        else:
            # Normal block
            blocks.append(make_function_block_bytes(rng, num_lines=rng.randint(8, 20)))
    return b"".join(blocks)


def stream_synthetic_shards(
    output_dir: str = "data/processed/shards",
    target_gb: float = 10.0,
    shard_size_mb: float = 100.0,
    avg_file_size_kb: int = 64,
    versions: list = None,
    seed: int = 42,
):
    """
    Streams synthetic objects directly into compressed Parquet shards.
    Keeps RAM usage strictly bounded to a single shard buffer (O(1) memory relative to total dataset size).
    """
    if versions is None:
        versions = ["synth_v1", "synth_v2", "synth_v3"]

    os.makedirs(output_dir, exist_ok=True)
    target_bytes = int(target_gb * (1024 ** 3))
    shard_bytes_target = int(shard_size_mb * (1024 * 1024))
    avg_file_bytes = avg_file_size_kb * 1024

    approx_total_objects = target_bytes // avg_file_bytes
    objects_per_version = max(10, approx_total_objects // len(versions))
    blocks_per_file = max(2, avg_file_bytes // 600)  # ~600 bytes per block

    print(f"=== Synthetic Data Generator (Target: {target_gb:.2f} GB) ===")
    print(f"Versions: {versions}")
    print(f"Estimated Objects: {approx_total_objects:,} ({objects_per_version:,} per version)")
    print(f"Target Shard Size: {shard_size_mb} MB (compressed .parquet)")

    with MemoryTracker(f"Generate {target_gb:.1f}GB Synthetic Stream"):
        shard_idx = 0
        current_shard_rows = []
        current_shard_bytes = 0
        total_generated_bytes = 0
        total_objects = 0

        # Master RNG for reproducibility
        master_rng = random.Random(seed)

        # Baseline object descriptors: store only lightweight (file_id, file_seed, blocks)
        # instead of actual byte strings to keep RAM strictly minimal
        base_descriptors = {}
        for file_idx in range(objects_per_version):
            file_id = f"module_{file_idx}.c"
            file_seed = master_rng.randint(0, 2**31 - 1)
            num_blocks = max(2, int(master_rng.gauss(blocks_per_file, blocks_per_file * 0.2)))
            base_descriptors[file_id] = (file_seed, num_blocks)

        for v_idx, version_name in enumerate(versions):
            print(f"--> Generating {version_name}...")
            v_rng = random.Random(seed + v_idx * 1000)

            # Determine files in this version
            if v_idx == 0:
                current_files = dict(base_descriptors)
            else:
                current_files = {}
                for fid, (fseed, n_blocks) in base_descriptors.items():
                    r = v_rng.random()
                    if r < 0.05:
                        continue  # 5% deleted in this version
                    elif r < 0.35:
                        # 30% lightly modified
                        current_files[fid] = (fseed, n_blocks, 0.25)
                    else:
                        # Unchanged
                        current_files[fid] = (fseed, n_blocks, 0.0)

                # Add 10% new files
                num_new = max(5, int(len(base_descriptors) * 0.10))
                for n_i in range(num_new):
                    new_fid = f"module_new_v{v_idx}_{n_i}.c"
                    new_seed = v_rng.randint(0, 2**31 - 1)
                    new_blocks = max(2, int(v_rng.gauss(blocks_per_file, blocks_per_file * 0.2)))
                    current_files[new_fid] = (new_seed, new_blocks, 0.0)

            # Generate payload and chunks for each file in version
            for fname, spec in current_files.items():
                if len(spec) == 2:
                    fseed, n_blocks = spec
                    mutation = 0.0
                else:
                    fseed, n_blocks, mutation = spec

                obj_rng = random.Random(fseed)
                data_bytes = generate_object_payload(obj_rng, n_blocks, mutation_ratio=mutation)
                obj_size = len(data_bytes)

                # Chunking
                chunk_boundaries = gear_chunks(data_bytes, avg_size=8192, min_size=2048, max_size=16384)
                chunk_hashes = [hash_chunk(data_bytes[s:e]) for (s, e) in chunk_boundaries]

                object_id = f"{version_name}::{fname}"
                current_shard_rows.append({
                    "object_id": object_id,
                    "version": version_name,
                    "relative_path": fname,
                    "full_path": os.path.join("data", "raw", version_name, fname),
                    "object_size": obj_size,
                    "chunk_hashes": chunk_hashes,
                    "num_chunks": len(chunk_hashes),
                })

                current_shard_bytes += obj_size
                total_generated_bytes += obj_size
                total_objects += 1

                # Flush shard when target size reached
                if current_shard_bytes >= shard_bytes_target:
                    shard_file = os.path.join(output_dir, f"chunked_objects_shard_{shard_idx:04d}.parquet")
                    shard_df = pd.DataFrame(current_shard_rows)
                    shard_df = optimize_dtypes(shard_df)
                    shard_df.to_parquet(shard_file, compression="snappy", index=False)
                    print(
                        f"    [Shard {shard_idx:04d}] Wrote {len(shard_df):,} objects "
                        f"({current_shard_bytes / (1024**2):.1f} MB payload) -> {shard_file}"
                    )
                    current_shard_rows.clear()
                    current_shard_bytes = 0
                    shard_idx += 1
                    gc.collect()

        # Flush any remaining rows
        if current_shard_rows:
            shard_file = os.path.join(output_dir, f"chunked_objects_shard_{shard_idx:04d}.parquet")
            shard_df = pd.DataFrame(current_shard_rows)
            shard_df = optimize_dtypes(shard_df)
            shard_df.to_parquet(shard_file, compression="snappy", index=False)
            print(
                f"    [Shard {shard_idx:04d}] Wrote {len(shard_df):,} objects "
                f"({current_shard_bytes / (1024**2):.1f} MB payload) -> {shard_file}"
            )
            shard_idx += 1
            current_shard_rows.clear()
            gc.collect()

        print(
            f"\nFinished generation: {total_objects:,} objects across {shard_idx} shards "
            f"({total_generated_bytes / (1024**3):.2f} GB total synthetic volume)."
        )
        return shard_idx


def generate_raw_files_scaled(base_dir="data/raw", num_files=200, num_versions=3, seed=42):
    """
    Backwards-compatible raw file generator for small runs or specific disk tests.
    Writes files in streaming writes directly to disk.
    """
    random.seed(seed)
    os.makedirs(base_dir, exist_ok=True)

    v1_dir = os.path.join(base_dir, "synth_v1")
    os.makedirs(v1_dir, exist_ok=True)
    file_seeds = {}

    for i in range(num_files):
        fname = f"module_{i}.c"
        fseed = random.randint(0, 2**31 - 1)
        file_seeds[fname] = fseed
        content = generate_object_payload(random.Random(fseed), num_blocks=8).decode("utf-8", errors="ignore")
        with open(os.path.join(v1_dir, fname), "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Created {num_files} raw files in {v1_dir}")

    prev_seeds = file_seeds
    for v in range(2, num_versions + 1):
        v_dir = os.path.join(base_dir, f"synth_v{v}")
        os.makedirs(v_dir, exist_ok=True)
        new_seeds = {}

        for fname, fseed in prev_seeds.items():
            if random.random() < 0.05:
                continue
            mutation = random.choice([0.0, 0.15, 0.3])
            content = generate_object_payload(random.Random(fseed), num_blocks=8, mutation_ratio=mutation).decode("utf-8", errors="ignore")
            new_seeds[fname] = fseed
            with open(os.path.join(v_dir, fname), "w", encoding="utf-8") as f:
                f.write(content)

        for i in range(max(5, int(num_files * 0.1))):
            fname = f"module_new_v{v}_{i}.c"
            nseed = random.randint(0, 2**31 - 1)
            new_seeds[fname] = nseed
            content = generate_object_payload(random.Random(nseed), num_blocks=8).decode("utf-8", errors="ignore")
            with open(os.path.join(v_dir, fname), "w", encoding="utf-8") as f:
                f.write(content)

        print(f"Created {len(new_seeds)} files in {v_dir}")
        prev_seeds = new_seeds


def main():
    parser = argparse.ArgumentParser(description="Scalable Synthetic Data Generator")
    parser.add_argument("--tier", choices=list(SIZE_TIERS.keys()), default=None,
                        help="Size tier preset (mini=50MB, small=2GB, medium=5GB, large=10GB, xlarge=20GB)")
    parser.add_argument("--target_gb", type=float, default=None,
                        help="Custom target size in GB (e.g. 10.0)")
    parser.add_argument("--shard_size_mb", type=float, default=100.0,
                        help="Target uncompressed payload size per Parquet shard (MB)")
    parser.add_argument("--output_dir", type=str, default="data/processed/shards",
                        help="Directory to save compressed parquet shards")
    parser.add_argument("--mode", choices=["sharded", "raw_files"], default="sharded",
                        help="Output mode: 'sharded' (compressed parquet shards) or 'raw_files'")
    parser.add_argument("--num_files", type=int, default=200,
                        help="File count for raw_files mode")
    args = parser.parse_args()

    if args.mode == "raw_files":
        generate_raw_files_scaled(num_files=args.num_files)
        return

    # Determine target_gb
    if args.target_gb is not None:
        target_gb = args.target_gb
    elif args.tier is not None:
        target_gb = SIZE_TIERS[args.tier]
    else:
        # Default tier: small/mini for quick test, or user specified
        target_gb = SIZE_TIERS["mini"]

    stream_synthetic_shards(
        output_dir=args.output_dir,
        target_gb=target_gb,
        shard_size_mb=args.shard_size_mb,
    )


if __name__ == "__main__":
    main()