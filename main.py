"""
Main Execution Script: Cloud Data Lifecycle Optimization Pipeline
Runs end-to-end pipeline on real local production datasets:
- azure_functions_2019
- gcc_versions
- kernel_versions
"""
import os
import sys
import time

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingestion.build_objects import walk_and_chunk, get_dataset_versions
from src.features.redundancy import compute_redundancy_scores
from src.features.access_overlay import overlay_access_patterns
from src.features.dependency_graph import build_dependency_graph
from src.features.cost_estimate import estimate_cost, STORAGE_COST_PER_GB_MONTH
from src.labels.generate_labels import assign_label
from src.training.split import time_based_split
from src.training.train_policy import train
from src.evaluation.evaluate import evaluate
import pandas as pd
import pickle


def run_pipeline():
    total_start = time.time()
    print("=" * 70)
    print("Cloud Data Lifecycle Pipeline: Production Dataset Execution")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Ingestion & FastCDC Chunking
    # -------------------------------------------------------------
    print("\n[Stage 1/9] Ingesting & Chunking Production Datasets...")
    discovered = get_dataset_versions()
    print(f"Discovered {len(discovered)} dataset version groups:")
    for vname, vpath in discovered.items():
        print(f"  - {vname}: {vpath}")

    chunked_df = walk_and_chunk(discovered, max_files=250)
    print(f"-> Ingested {len(chunked_df)} total objects across {chunked_df['version'].nunique()} versions.", flush=True)
    print(f"   Total raw bytes represented: {chunked_df['object_size'].sum():,} bytes", flush=True)

    # -------------------------------------------------------------
    # Step 2: Redundancy Analysis
    # -------------------------------------------------------------
    print("\n[Stage 2/9] Computing Multi-Version Redundancy Scores...")
    redundancy_df = compute_redundancy_scores(chunked_df)
    redundancy_path = os.path.join("data", "processed", "objects_with_redundancy.parquet")
    os.makedirs(os.path.dirname(redundancy_path), exist_ok=True)
    redundancy_df.to_parquet(redundancy_path)
    mean_redundancy = redundancy_df["redundancy_score"].mean()
    print(f"-> Mean object redundancy score across corpus: {mean_redundancy:.4f}")

    # -------------------------------------------------------------
    # Step 3: Access Pattern Overlay (Azure Functions 2019)
    # -------------------------------------------------------------
    print("\n[Stage 3/9] Applying Access Pattern Features (Empirical Azure Trace)...")
    access_df = overlay_access_patterns(redundancy_df)
    access_path = os.path.join("data", "processed", "objects_with_access.parquet")
    os.makedirs(os.path.dirname(access_path), exist_ok=True)
    access_df.to_parquet(access_path)
    print(f"-> Access feature summary (mean counts):")
    print(f"   7-day: {access_df['access_freq_7d'].mean():.1f} | 30-day: {access_df['access_freq_30d'].mean():.1f} | 90-day: {access_df['access_freq_90d'].mean():.1f}")

    # -------------------------------------------------------------
    # Step 4: Dependency Graph Analysis
    # -------------------------------------------------------------
    print("\n[Stage 4/9] Building C/C++ Header Dependency DAG...")
    G = build_dependency_graph(access_df)
    in_degrees = dict(G.in_degree())
    access_df["num_dependents"] = access_df["object_id"].map(lambda oid: in_degrees.get(oid, 0))
    deps_path = os.path.join("data", "processed", "objects_with_deps.parquet")
    os.makedirs(os.path.dirname(deps_path), exist_ok=True)
    access_df.to_parquet(deps_path)
    with open(os.path.join("data", "processed", "dependency_graph.gpickle"), "wb") as f:
        pickle.dump(G, f)
    print(f"-> Dependency Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Max dependents on a single object: {access_df['num_dependents'].max()}")

    # -------------------------------------------------------------
    # Step 5: Cost & Compression Estimation
    # -------------------------------------------------------------
    print("\n[Stage 5/9] Estimating Compression Ratio & Cloud Storage Costs...")
    results = access_df["full_path"].apply(estimate_cost)
    access_df["compression_ratio"] = results.apply(lambda r: r["compression_ratio"] if r else 1.0)
    access_df["compute_time_sec"] = results.apply(lambda r: r["compute_time_sec"] if r else 0.0)
    access_df["compressed_size_bytes"] = results.apply(lambda r: r["compressed_size_bytes"] if r else 0)
    access_df["monthly_storage_cost_usd"] = (access_df["object_size"] / (1024 ** 3)) * STORAGE_COST_PER_GB_MONTH["hot"]
    cost_path = os.path.join("data", "processed", "objects_with_cost.parquet")
    os.makedirs(os.path.dirname(cost_path), exist_ok=True)
    access_df.to_parquet(cost_path)
    print(f"-> Mean compression ratio: {access_df['compression_ratio'].mean():.2f}x")
    print(f"   Total uncompressed monthly baseline cost: ${access_df['monthly_storage_cost_usd'].sum():.6f}")

    # -------------------------------------------------------------
    # Step 6: Lifecycle Label Generation
    # -------------------------------------------------------------
    print("\n[Stage 6/9] Generating Ground Truth Lifecycle Labels...")
    access_df["lifecycle_label"] = access_df.apply(assign_label, axis=1)
    labels_path = os.path.join("data", "labels", "objects_dataset.parquet")
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    access_df.to_parquet(labels_path)
    print("-> Distribution of assigned lifecycle actions:")
    for action, count in access_df["lifecycle_label"].value_counts().items():
        print(f"   {action.upper():<10}: {count} objects ({count/len(access_df)*100:.1f}%)")

    # -------------------------------------------------------------
    # Step 7: Chronological Train / Test Split
    # -------------------------------------------------------------
    print("\n[Stage 7/9] Performing Time-Based Chronological Split...")
    train_df, test_df = time_based_split(access_df)
    train_path = os.path.join("data", "labels", "train.parquet")
    test_path = os.path.join("data", "labels", "test.parquet")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    print(f"-> Train Set: {len(train_df)} objects ({train_df['version'].nunique()} versions: {list(train_df['version'].unique())})")
    print(f"-> Test Set:  {len(test_df)} objects ({test_df['version'].nunique()} versions: {list(test_df['version'].unique())})")

    # -------------------------------------------------------------
    # Step 8: XGBoost Model Training
    # -------------------------------------------------------------
    print("\n[Stage 8/9] Training Policy Classifier (XGBoost)...")
    train()

    # -------------------------------------------------------------
    # Step 9: Policy Cost & Savings Evaluation
    # -------------------------------------------------------------
    print("\n[Stage 9/9] Evaluating Lifecycle Policy Savings on Held-Out Test Set...")
    evaluate()

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"Pipeline completed successfully in {elapsed:.2f} seconds!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
