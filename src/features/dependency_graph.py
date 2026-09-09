"""
Builds a dependency graph by parsing #include statements in C/C++ files.
This gives REAL dependency structure instead of a random graph.
"""
import os
import re
import networkx as nx
import pandas as pd
import pickle

INCLUDE_RE = re.compile(r'#include\s*[<"]([^>"]+)[>"]')


def extract_includes(filepath: str):
    includes = []
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

    # Map basename -> object_id for resolving includes (approximate matching -- documented limitation)
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

    return G


if __name__ == "__main__":
    in_path = os.path.join("data", "processed", "objects_with_access.parquet")
    df = pd.read_parquet(in_path)

    G = build_dependency_graph(df)

    # num_dependents = in-degree (how many other objects depend on this one)
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