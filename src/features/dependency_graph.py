"""
Builds a dependency graph by parsing #include statements in C/C++ files
or synthesizes dependency links for virtual synthetic workloads.
Memory-optimized with downcasting to uint16/int16.
"""
import os
import sys
import re
import networkx as nx
import pandas as pd
import pickle
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.memory_opt import optimize_dtypes, MemoryTracker

INCLUDE_RE = re.compile(r'#include\s*[<"]([^>"]+)[>"]')


def extract_includes(filepath: str):
    includes = []
    if not os.path.exists(filepath):
        return includes
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = INCLUDE_RE.match(line.strip())
                if m:
                    includes.append(os.path.basename(m.group(1)))
    except Exception:
        pass
    return includes


def build_dependency_graph(df: pd.DataFrame, seed: int = 42):
    G = nx.DiGraph()
    rng = random.Random(seed)

    first_file = df["full_path"].iloc[0] if not df.empty else ""
    has_real_files = os.path.exists(first_file)

    if has_real_files:
        basename_to_id = {}
        for _, row in df.iterrows():
            basename_to_id.setdefault(os.path.basename(row["relative_path"]), row["object_id"])

        for _, row in df.iterrows():
            G.add_node(row["object_id"])
            includes = extract_includes(row["full_path"])
            for inc in includes:
                target_id = basename_to_id.get(inc)
                if target_id and target_id != row["object_id"]:
                    G.add_edge(row["object_id"], target_id)
    else:
        # Fast path for synthetic datasets
        all_ids = df["object_id"].tolist()
        G.add_nodes_from(all_ids)
        num_links = min(50000, int(len(all_ids) * 0.3))
        for _ in range(num_links):
            src = rng.choice(all_ids)
            tgt = rng.choice(all_ids)
            if src != tgt:
                G.add_edge(src, tgt)

    return G


def process_dependencies(
    in_path: str = "data/processed/objects_with_access.parquet",
    out_path: str = "data/processed/objects_with_deps.parquet",
    graph_out: str = "data/processed/dependency_graph.gpickle",
):
    with MemoryTracker("Build Dependency Graph"):
        df = pd.read_parquet(in_path)
        G = build_dependency_graph(df)

        in_degrees = dict(G.in_degree())
        df["num_dependents"] = df["object_id"].map(lambda oid: in_degrees.get(oid, 0)).astype("uint16")

        df = optimize_dtypes(df, verbose=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, compression="snappy", index=False)

        os.makedirs(os.path.dirname(graph_out), exist_ok=True)
        with open(graph_out, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Graph has {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
        print(f"Saved {len(df):,} objects with dependencies to {out_path}")
    return df


if __name__ == "__main__":
    process_dependencies()
