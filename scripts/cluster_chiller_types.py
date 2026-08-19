"""
Derives chiller instrumentation "types" from data/trend_wide.csv by clustering
chillers on which sensor columns they actually populate — never hardcoded,
per CLAUDE.md ("types must be discovered fresh from the data ... since the
fleet may have changed"). Read-only: only reads the CSV, no DB access, no
mutation of the source file.

This does NOT model sensor values themselves, only presence/absence, so the
per-chiller / never-pooled modeling rule in CLAUDE.md doesn't apply here --
that rule is about fitting predictive models on readings, not about grouping
chillers by which columns they log.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

METADATA_COLUMNS = {"machineId", "timestamp", "status", "Criticality"}
POPULATED_THRESHOLD = 0.05  # a column counts as "populated" for a chiller if
                             # more than 5% of its rows have a non-null value
CANDIDATE_K = [2, 3, 4]


def load_coverage(df):
    """Per-machineId fraction of non-null rows for each sensor column."""
    sensor_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    coverage = df.groupby("machineId")[sensor_cols].apply(lambda g: g.notna().mean())
    return coverage, sensor_cols


def pick_best_k(populated):
    """Try each candidate k, return (k, labels) with the best silhouette score."""
    best_k, best_labels, best_score = None, None, -1.0
    n_samples = len(populated)

    for k in CANDIDATE_K:
        if k >= n_samples:
            continue
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(populated.values)
        score = silhouette_score(populated.values, labels)
        print(f"  k={k}: silhouette score = {score:.3f}")
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score

    return best_k, best_labels, best_score


def label_clusters_by_size(populated, raw_labels):
    """Relabel clusters as type_1, type_2, ... ordered by descending avg columns populated (mirrors CLAUDE.md's ~13/22/24 ordering)."""
    n_populated_per_chiller = populated.sum(axis=1)
    cluster_avg_columns = (
        pd.Series(n_populated_per_chiller.values, index=raw_labels)
        .groupby(level=0)
        .mean()
        .sort_values(ascending=False)
    )
    raw_to_name = {raw: f"type_{i + 1}" for i, raw in enumerate(cluster_avg_columns.index)}
    return pd.Series(raw_labels, index=populated.index).map(raw_to_name)


def summarize_clusters(populated, chiller_types, sensor_cols):
    print("\nCluster summary:")
    for type_name in sorted(chiller_types.unique()):
        members = chiller_types[chiller_types == type_name].index
        cluster_pop = populated.loc[members]
        avg_columns = cluster_pop.sum(axis=1).mean()
        defining_cols = [c for c in sensor_cols if cluster_pop[c].mean() >= 0.8]

        print(f"\n  {type_name}: {len(members)} chillers, avg {avg_columns:.1f} columns populated")
        print(f"    machineIds: {sorted(members.tolist())}")
        print(f"    defining columns ({len(defining_cols)}): {defining_cols}")


def main():
    parser = argparse.ArgumentParser(description="Cluster chillers into instrumentation types by column coverage.")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "trend_wide.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "chiller_types.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    coverage, sensor_cols = load_coverage(df)
    populated = (coverage > POPULATED_THRESHOLD).astype(int)

    print(f"{len(populated)} chillers, {len(sensor_cols)} candidate sensor columns.")
    print("Trying candidate cluster counts:")
    best_k, best_labels, best_score = pick_best_k(populated)
    print(f"\nSelected k={best_k} (silhouette score = {best_score:.3f})")

    chiller_types = label_clusters_by_size(populated, best_labels)
    summarize_clusters(populated, chiller_types, sensor_cols)

    meta = df.groupby("machineId")[["status", "Criticality"]].first()

    out = pd.concat(
        [chiller_types.rename("chiller_type"), meta, populated.astype(bool)],
        axis=1,
    ).reset_index()

    out.to_csv(args.output, index=False)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
