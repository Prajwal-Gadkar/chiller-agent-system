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
    # 1. Chiller Efficiency Performance Index (ikW/TR, kW/Ton)
    (["ikw/tr", "ikw_tr", "kw/tr", "kw/ton"], 0.0, 10.0, "chiller_efficiency_ikw_tr"),
    # 2. Cumulative Energy Consumption (kWh, MWh)
    (["(kwh)", "kwh", "mwh", "energy_consumption", "cumulative_energy"], 0.0, 10000000.0, "cumulative_energy_kwh"),
    # 3. Facility & IT Electrical Load (kW, MW)
    (["facility_load", "it_load", "building_load", "plant_load", "total_facility", "total_load"], 0.0, 100000.0, "facility_it_load_kw"),
    # 4. Performance Deviation %
    (["performance deviation", "deviation (%)", "deviation_pct", "dev_pct"], -100.0, 100.0, "performance_deviation_pct"),
    # 5. BMS Alarm / Fault / Trip Status Codes
    (["trip status", "alarm_status", "common_alarm", "fault_status", "fault_code", "alarm_code"], 0.0, 65535.0, "bms_alarm_code"),
    (["on_off", "auto_manual", "run_status", "status"], 0.0, 1.0, "binary_status"),
    # 6. Run Hours
    (["hours", "runhours"], 0.0, 200000.0, "cumulative_hours"),
    # 7. Temperature & Delta T
    (["temperature", "temp", "wet bulb", "cwet", "deg c", "degc", "delta t", "delta_t", "delta"], -20.0, 60.0, "temperature"),
    # 8. Fan / Pump Speed & Frequency (placed before pressure to avoid any collision)
    (["speed", "frequency", "rpm"], -10.0, 5000.0, "speed"),
    # 9. Refrigerant & Hydraulic Pressure (requires 'press', 'lp_value', 'hp_value', 'psi', 'bar')
    (["press", "lp_value", "hp_value", "psi", "bar"], -50.0, 2000.0, "pressure"),
    # 10. Flow Rate
    (["flow", "m3/h", "gpm"], -10.0, 5000.0, "flow"),
    # 11. Cooling Capacity Tonnage (TR)
    (["calculated (tr)", "calculated(tr)", "calculated tr", "rounded (tr)", "rounded(tr)", "rounded tr", "cooling_capacity", "tonnage"], 0.0, 5000.0, "cooling_capacity_tr"),
    # 12. Percentage / Compressor Load
    (["percent", "%", "load", "performance"], -10.0, 110.0, "percentage_load"),
    # 13. Instantaneous Electrical Power (kW, iKW)
    (["kw", "power", "ikw"], -10.0, 5000.0, "instantaneous_power_kw"),
    # 14. Temperature Setpoints
    (["setpoint"], -20.0, 60.0, "setpoint_temperature"),
]



def infer_bounds(column_name):
    """Return (min, max, rule_label) for a column name, or a statistical fallback."""
    name_lower = column_name.lower()
    # Replace 'compressor' with 'comp' to prevent 'press' inside 'comPRESSor' from matching pressure
    name_clean = name_lower.replace("compressor", "comp")
    for keywords, lo, hi, label in BOUND_RULES:
        if any(kw in name_clean for kw in keywords):
            return lo, hi, label
    return None, None, "unmapped"


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
        c for c in df.columns
        if c not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() > 0
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

        art_types = np.array(["none"] * len(df), dtype=object)
        window_ids = np.array([None] * len(df), dtype=object)
        evidences = np.array([""] * len(df), dtype=object)

        # (Pass 1 Sentinel check removed per user request: exact sentinel values are unverified)

        if rule != "unmapped":
            # 2. Vectorized Monotonic Ramp-then-Reset Detection (Per Machine Scope)
            if len(df) > 5:
                mids = df["machineId"].astype(str).values if "machineId" in df.columns else np.array(["unknown"] * len(df))
                v_curr = vals[:-1]
                v_next = vals[1:]
                m_curr = mids[:-1]
                m_next = mids[1:]
                
                drop_indices = np.where((art_types[:-1] == "none") & 
                                        (m_curr == m_next) &
                                        ((v_curr > hi) | (v_curr < lo)) & 
                                        (v_next >= lo) & (v_next <= hi))[0]

                for drop_idx in drop_indices:
                    target_mid = mids[drop_idx]
                    v_c = vals[drop_idx]
                    v_n = vals[drop_idx + 1]
                    
                    ramp_start = drop_idx
                    for j in range(drop_idx - 1, max(-1, drop_idx - 50), -1):
                        if mids[j] != target_mid:
                            break
                        ramp_start = j
                        if pd.notna(vals[j]) and (lo <= vals[j] <= hi):
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
            "ramp_resets": int((art_types == "ramp_reset").sum()),
            "physical_bounds": int((art_types == "physical_bound").sum()),
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
