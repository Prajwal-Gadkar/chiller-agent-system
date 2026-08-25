"""
agents/data_validation.py

Artifact-Aware Data Validation Agent — sanity, sentinel, and pattern gate on
every reading before downstream agents process it.

Read-only on source data: never drops or mutates rows, only annotates them with
companion metadata columns:
  - <col>_artifact_type: physical_bound | statistical_bound | sentinel | ramp_reset | genuine_anomaly | unclassified | none
  - <col>_fault_window_id: string event identifier (groups sustained window under one ID)
  - <col>_evidence: short human/LLM-readable rationale

Wired directly into live knowledge store (chiller_agent_knowledge.json) via knowledge_store.py.
"""

import os
import sys
import uuid
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import knowledge_store

METADATA_COLUMNS = {"machineId", "timestamp", "status", "Criticality"}

BOUND_RULES = [
    (["status", "alarm", "fault", "trip"], 0.0, 1.0, "duty_cycle_status"),
    (["percent", "%", "deviation", "load", "performance"], -10.0, 110.0, "percentage"),
    (["hours", "runhours"], 0.0, 200000.0, "cumulative_hours"),
    (["temperature", "temp", "wet bulb", "cwet"], -20.0, 60.0, "temperature"),
    (["pressure", "press"], -50.0, 500.0, "pressure"),
    (["flow"], -10.0, 5000.0, "flow"),
    (["speed"], -10.0, 110.0, "speed_pct"),
    (["kw", "power", "ikw"], -10.0, 5000.0, "power"),
    (["setpoint"], -20.0, 60.0, "setpoint_temperature"),
]

STATISTICAL_FALLBACK_SIGMA = 6


def infer_bounds(column_name):
    """Return (min, max, rule_label) for a column name, or a statistical fallback."""
    name_lower = column_name.lower()
    name_check = name_lower.replace("compressor", "comp")
    for keywords, lo, hi, label in BOUND_RULES:
        if any(kw in name_check for kw in keywords):
            return lo, hi, label
    return None, None, "statistical_fallback"


def get_validation_agent_prompt(df: pd.DataFrame = None) -> str:
    """Return the compact LLM prompt context bundle from the live knowledge store."""
    return knowledge_store.get_agent_context()


def load_sentinel_values() -> list:
    """Load confirmed sentinel values from knowledge store."""
    kb_data = knowledge_store.load()
    return kb_data.get("validation_rules", {}).get("sentinel_values", {}).get("values", [0, -1, 9999, 10000, 65535, -999])


def validate(df: pd.DataFrame):
    """
    Apply multi-layer artifact-aware validation to a wide-format DataFrame.

    Returns (annotated_df, report_df):
      - annotated_df: copy of df with companion metadata columns per sensor:
          - <col>_artifact_type
          - <col>_fault_window_id
          - <col>_evidence
      - report_df: summary table per validated column with percentage breakdown.
    """
    annotated_df = df.copy()
    sentinel_vals = load_sentinel_values()
    kb_data = knowledge_store.load()
    cluster_entries = kb_data.get("cluster_registry", {}).get("entries", {})

    value_columns = [
        c for c in df.columns if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]

    companion_cols = {}
    report_rows = []

    mid_to_cluster = {}
    for cid, centry in cluster_entries.items():
        for m in centry.get("members", []):
            mid_to_cluster[str(m)] = cid

    df_mids = df["machineId"].astype(str).values if "machineId" in df.columns else np.array(["unknown"] * len(df))
    df_clusters = np.array([mid_to_cluster.get(m) for m in df_mids])

    for col in value_columns:
        series = df[col]
        vals = series.values
        lo, hi, rule = infer_bounds(col)

        if rule == "statistical_fallback":
            mean = series.mean()
            std = series.std()
            lo = mean - STATISTICAL_FALLBACK_SIGMA * std if pd.notna(mean) else -9999
            hi = mean + STATISTICAL_FALLBACK_SIGMA * std if pd.notna(mean) else 9999

        art_types = np.array(["none"] * len(df), dtype=object)
        window_ids = np.array([None] * len(df), dtype=object)
        evidences = np.array([""] * len(df), dtype=object)

        # 1. Sentinel Value Check
        for sentinel in sentinel_vals:
            if sentinel == 0:
                if any(kw in col.lower() for kw in ["kw", "flow", "speed", "power", "load"]):
                    is_z = (vals == 0)
                    flat4 = is_z & np.roll(is_z, 1) & np.roll(is_z, 2) & np.roll(is_z, 3)
                    idx_sent = np.where(flat4)[0]
                    art_types[idx_sent] = "sentinel"
                    evidences[idx_sent] = "Sustained zero flatline on dynamic parameter"
            else:
                sent_mask = np.isclose(vals, sentinel, atol=0.01, equal_nan=False)
                idx_sent = np.where(sent_mask)[0]
                art_types[idx_sent] = "sentinel"
                evidences[idx_sent] = f"Exact sentinel value match ({sentinel})"

        # 2. Vectorized Monotonic Ramp-then-Reset Detection
        if len(df) > 5:
            v_curr = vals[:-1]
            v_next = vals[1:]
            
            drop_indices = np.where((art_types[:-1] == "none") & 
                                    ((v_curr > hi) | (v_curr < lo)) & 
                                    (v_next >= lo) & (v_next <= hi))[0]

            for drop_idx in drop_indices:
                v_c = vals[drop_idx]
                v_n = vals[drop_idx + 1]
                
                ramp_start = max(0, drop_idx - 10)
                for j in range(drop_idx - 1, max(0, drop_idx - 50), -1):
                    if pd.notna(vals[j]) and (lo <= vals[j] <= hi):
                        ramp_start = j
                        break

                f_window = f"fw-ramp-{uuid.uuid4().hex[:6]}"
                evidence_str = f"Monotonic ramp-reset window: peak {v_c:.2f} -> drop to {v_n:.2f}"

                art_types[ramp_start:drop_idx + 1] = "ramp_reset"
                window_ids[ramp_start:drop_idx + 1] = f_window
                evidences[ramp_start:drop_idx + 1] = evidence_str

        # 3. Layer 1 Physical Bound Check
        phys_mask = ((vals < lo) | (vals > hi)) & (art_types == "none") & pd.notna(vals)
        idx_phys = np.where(phys_mask)[0]
        art_types[idx_phys] = "physical_bound"
        for idx in idx_phys:
            evidences[idx] = f"Physical bound breach (min: {lo}, max: {hi}, value: {vals[idx]:.2f})"

        # 4. Vectorized Layer 2 Statistical Bound Check (per cluster)
        if "machineId" in df.columns:
            unassigned_mask = (art_types == "none") & pd.notna(vals)
            for cid, centry in cluster_entries.items():
                sbounds = centry.get("stat_bounds", {}).get(col)
                if sbounds:
                    p1, p99 = sbounds["p1"], sbounds["p99"]
                    c_mask = unassigned_mask & (df_clusters == cid) & ((vals < p1) | (vals > p99))
                    idx_stat = np.where(c_mask)[0]
                    art_types[idx_stat] = "statistical_bound"
                    for idx in idx_stat:
                        evidences[idx] = f"Layer-2 statistical bound breach (cluster {cid} p1: {p1}, p99: {p99}, value: {vals[idx]:.2f})"

        companion_cols[f"{col}_artifact_type"] = art_types
        companion_cols[f"{col}_fault_window_id"] = window_ids
        companion_cols[f"{col}_evidence"] = evidences

        n_total = int(series.notna().sum())
        n_flagged = int((art_types != "none").sum())
        pct_flagged = (n_flagged / n_total * 100) if n_total > 0 else np.nan

        report_rows.append({
            "column": col,
            "rule": rule,
            "bound_min": lo,
            "bound_max": hi,
            "n_total": n_total,
            "n_flagged": n_flagged,
            "pct_flagged": pct_flagged,
            "sentinels": int((art_types == "sentinel").sum()),
            "ramp_resets": int((art_types == "ramp_reset").sum()),
            "physical_bounds": int((art_types == "physical_bound").sum()),
            "statistical_bounds": int((art_types == "statistical_bound").sum()),
            "genuine_anomalies": int((art_types == "genuine_anomaly").sum()),
        })

    comp_df = pd.DataFrame(companion_cols, index=df.index)
    annotated_df = pd.concat([df.copy(), comp_df], axis=1)

    report_df = pd.DataFrame(report_rows).sort_values("pct_flagged", ascending=False).reset_index(drop=True)
    return annotated_df, report_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the artifact-aware data validation agent.")
    parser.add_argument("csv_path", help="Path to a wide-format trend CSV (e.g. data/trend_wide.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    _, report_df = validate(df)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
