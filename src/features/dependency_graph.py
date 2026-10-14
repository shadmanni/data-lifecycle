"""
Builds a dependency graph:
- For CMaps: models telemetry dependencies between train baselines, test runs, RUL ground truths, and engine partitions.
- For C/C++ source code: parses #include statements.
"""
import os
import re
import networkx as nx
import pandas as pd
import pickle

INCLUDE_RE = re.compile(r'#include\s*[<"]([^>"]+)[>"]')


def extract_includes(filepath: str):
    includes = []
    if not os.path.isfile(filepath):
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


def build_dependency_graph(df: pd.DataFrame):
    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_node(row["object_id"])

    # Basename lookup for include statements
    basename_to_id = {}
    for _, row in df.iterrows():
        basename_to_id.setdefault(os.path.basename(row["relative_path"]), row["object_id"])

    # 1. C/C++ Header Dependency Check
    for _, row in df.iterrows():
        includes = extract_includes(row["full_path"])
        for inc in includes:
            target_id = basename_to_id.get(inc)
            if target_id and target_id != row["object_id"]:
                G.add_edge(row["object_id"], target_id)

    # 2. CMaps Telemetry Dependency Structure
    # Subsets: FD001, FD002, FD003, FD004
    for version in df["version"].unique():
        v_df = df[df["version"] == version]
        v_oids = v_df["object_id"].tolist()

        train_master = next((oid for oid in v_oids if f"train_{version}" in oid or "::train_" in oid), None)
        test_master = next((oid for oid in v_oids if f"test_{version}" in oid or "::test_" in oid), None)
        rul_master = next((oid for oid in v_oids if f"RUL_{version}" in oid or "::RUL_" in oid or "rul::groundtruth" in oid), None)

        # Test set depends on Training baseline
        if test_master and train_master and test_master != train_master:
            G.add_edge(test_master, train_master)

        # RUL groundtruth depends on Test engines
        if rul_master and test_master and rul_master != test_master:
            G.add_edge(rul_master, test_master)

        # Engine unit partitions depend on their master dataset
        for oid in v_oids:
            if "::train::" in oid and train_master and oid != train_master:
                G.add_edge(oid, train_master)
            elif "::test::" in oid and test_master and oid != test_master:
                G.add_edge(oid, test_master)

    return G


if __name__ == "__main__":
    in_path = os.path.join("data", "processed", "objects_with_access.parquet")
    df = pd.read_parquet(in_path)

    G = build_dependency_graph(df)

    in_degrees = dict(G.in_degree())
    df["num_dependents"] = df["object_id"].map(lambda oid: in_degrees.get(oid, 0))

    out_path = os.path.join("data", "processed", "objects_with_deps.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)

    with open(os.path.join("data", "processed", "dependency_graph.gpickle"), "wb") as f:
        pickle.dump(G, f)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(df["num_dependents"].describe())
    print(f"Saved to {out_path}")