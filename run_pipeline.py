"""
End-to-End Scalable Data Lifecycle Pipeline Orchestrator.
Runs data generation, feature engineering, labeling, training, and evaluation
with memory optimization, assertion hooks, and streaming disk shards.
"""
import os
import sys
import time
import argparse
import logging

# Ensure project root in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ingestion.generate_synthetic_data import stream_synthetic_shards, SIZE_TIERS
from src.features.redundancy import process_dataset as process_redundancy
from src.features.access_overlay import overlay_access_patterns
from src.features.dependency_graph import process_dependencies
from src.features.cost_estimate import process_costs
from src.labels.generate_labels import generate_labels
from src.training.split import run_split
from src.training.train_policy import train as run_train
from src.evaluation.evaluate import evaluate as run_eval
from src.utils.memory_opt import MemoryTracker, get_memory_info, log_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pipeline_orchestrator")


def run_pipeline(
    tier: str = "mini",
    target_gb: float = None,
    model_type: str = "lightgbm",
    shard_size_mb: float = 100.0,
    skip_data_gen: bool = False,
    batch_size: int = 5000,
):
    start_total_time = time.time()

    if target_gb is None:
        target_gb = SIZE_TIERS.get(tier, 0.05)

    print("================================================================================")
    print("        SCALABLE CLOUD DATA LIFECYCLE MANAGEMENT PIPELINE")
    print("================================================================================")
    mem_info = get_memory_info()
    print(f"Target Scale Volume : {target_gb:.2f} GB (Tier: '{tier}')")
    print(f"Model Backend       : {model_type.upper()}")
    print(f"System Available RAM: {mem_info['system_available_gb']:.2f} GB ({mem_info['system_percent']}% in use)")
    print("================================================================================\n")

    # Stage 1: Synthetic Data Scaling (Streaming Shards)
    if not skip_data_gen:
        logger.info(f"[STAGE 1/8] Generating Synthetic Dataset Stream (Target: {target_gb:.2f} GB)...")
        stream_synthetic_shards(
            output_dir="data/processed/shards",
            target_gb=target_gb,
            shard_size_mb=shard_size_mb,
        )
    else:
        logger.info("[STAGE 1/8] Skipping data generation (--skip_data_gen active).")

    # Stage 2: Redundancy Scoring & Deduplication Analysis
    logger.info("[STAGE 2/8] Computing Redundancy Scores across Chunk Hashes...")
    process_redundancy(
        in_path="data/processed/chunked_objects.parquet",
        shards_dir="data/processed/shards",
        out_path="data/processed/objects_with_redundancy.parquet",
        drop_chunk_hashes=True,  # drop heavy hash string arrays to free RAM
    )

    # Stage 3: Access Pattern & Recency Overlay
    logger.info("[STAGE 3/8] Simulating Zipfian Access Skew and Recency Features...")
    import pandas as pd
    df_red = pd.read_parquet("data/processed/objects_with_redundancy.parquet")
    df_acc = overlay_access_patterns(df_red)
    df_acc.to_parquet("data/processed/objects_with_access.parquet", compression="snappy", index=False)
    del df_red, df_acc

    # Stage 4: Dependency Graph Analysis
    logger.info("[STAGE 4/8] Building Object Dependency Graph & In-Degree Metrics...")
    process_dependencies(
        in_path="data/processed/objects_with_access.parquet",
        out_path="data/processed/objects_with_deps.parquet",
    )

    # Stage 5: Compression & Cloud Storage Cost Estimation
    logger.info("[STAGE 5/8] Estimating Compression Ratios and Tier Storage Costs...")
    process_costs(
        in_path="data/processed/objects_with_deps.parquet",
        out_path="data/processed/objects_with_cost.parquet",
    )

    # Stage 6: Ground-Truth Policy Labeling & Assertions
    logger.info("[STAGE 6/8] Generating Ground-Truth Lifecycle Policy Labels...")
    generate_labels(
        in_path="data/processed/objects_with_cost.parquet",
        out_path="data/labels/objects_dataset.parquet",
    )

    # Stage 7: Time-Based Train/Test Splitting
    logger.info("[STAGE 7/8] Executing Leakage-Free Train/Test Split...")
    run_split(
        in_path="data/labels/objects_dataset.parquet",
        train_out="data/labels/train.parquet",
        test_out="data/labels/test.parquet",
    )

    # Stage 8: Model Training with Assertion Checks & Memory Hooks
    logger.info(f"[STAGE 8/8] Training Policy Model ({model_type.upper()})...")
    clf, le, f1 = run_train(
        model_type=model_type,
        train_path="data/labels/train.parquet",
        test_path="data/labels/test.parquet",
        models_dir="models",
        batch_size=batch_size,
    )

    # Evaluation
    logger.info("Executing Storage Cost Savings Evaluation...")
    run_eval()

    elapsed = time.time() - start_total_time
    print("\n================================================================================")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s | Final Model: {model_type.upper()} (F1: {f1:.4f})")
    print("================================================================================")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Scalable Lifecycle Pipeline")
    parser.add_argument("--tier", choices=list(SIZE_TIERS.keys()), default="mini",
                        help="Data size tier: mini (50MB), small (2GB), medium (5GB), large (10GB), xlarge (20GB)")
    parser.add_argument("--target_gb", type=float, default=None,
                        help="Explicit target size in GB (e.g. 10.0)")
    parser.add_argument("--model_type", choices=["lightgbm", "xgboost", "incremental"], default="lightgbm",
                        help="Training backend: 'lightgbm' (hist CPU), 'xgboost' (hist), or 'incremental' (partial_fit)")
    parser.add_argument("--shard_size_mb", type=float, default=100.0,
                        help="Target size per compressed Parquet shard in MB")
    parser.add_argument("--skip_data_gen", action="store_true",
                        help="Skip synthetic data generation if already generated")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="Batch size for incremental learning and shard processing")
    args = parser.parse_args()

    run_pipeline(
        tier=args.tier,
        target_gb=args.target_gb,
        model_type=args.model_type,
        shard_size_mb=args.shard_size_mb,
        skip_data_gen=args.skip_data_gen,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
