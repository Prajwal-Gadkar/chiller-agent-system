"""
LangGraph Pipeline for Chiller Agent System.

Wires together validated components into an end-to-end execution graph:
1. State Schema: PipelineState (chiller_id, timestamp, raw_reading, validation_result, anomaly_result, insight_text)
2. Node validate_reading: Wraps agents/data_validation.py validate() for a single incoming reading.
3. Node check_anomaly: Loads per-chiller trained model from data/anomaly_models/, predicts expected KW, computes z-score & is_anomaly (|z| > 3).
4. Conditional Edge: if is_anomaly is True -> route to generate_insight; if False -> route to log_normal.
5. Node generate_insight: Placeholder NLG node formatting operational insight string.
6. Node log_normal: Formats normal operational state log string.
7. Test Harness: Pulls 20 real recent readings for 2-3 chillers from the database and runs them through the pipeline.
"""

import os
import sys
import pickle
import warnings
import pandas as pd
import numpy as np
import psycopg2
from typing import Dict, Any, Optional, TypedDict, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate
from agents.anomaly_agent import AnomalyAgent, MODEL_SAVE_DIR, find_sensor_column

load_dotenv()

# --- Load Asset Aliases Dedup Mapping ---
ALIAS_FILE = os.path.join(PROJECT_ROOT, "data", "asset_aliases.csv")
ASSET_ALIASES: Dict[int, Dict[str, Any]] = {}
if os.path.exists(ALIAS_FILE):
    try:
        alias_df = pd.read_csv(ALIAS_FILE)
        for _, row in alias_df.iterrows():
            alias_id = int(row["alias_id"])
            canonical_id = int(row["canonical_id"])
            name = str(row.get("canonical_name", f"Chiller-{canonical_id}"))
            ASSET_ALIASES[alias_id] = {
                "canonical_id": canonical_id,
                "canonical_name": name,
                "is_aliased": (alias_id != canonical_id)
            }
    except Exception as e:
        print(f"[Warning] Could not load asset aliases: {e}")


# --- 1. State Schema ---
class PipelineState(TypedDict):
    chiller_id: int
    timestamp: str
    raw_reading: Dict[str, Any]
    validation_result: Dict[str, Any]
    anomaly_result: Dict[str, Any]
    insight_text: Optional[str]


# --- Model Cache ---
_MODEL_CACHE: Dict[int, AnomalyAgent] = {}


def get_anomaly_agent(chiller_id: int) -> AnomalyAgent:
    """Retrieve or lazily load the trained AnomalyAgent model for a given chiller."""
    if chiller_id not in _MODEL_CACHE:
        _MODEL_CACHE[chiller_id] = AnomalyAgent.load(chiller_id, save_dir=MODEL_SAVE_DIR)
    return _MODEL_CACHE[chiller_id]


# --- 2. Node: validate_reading ---
def validate_reading(state: PipelineState) -> Dict[str, Any]:
    """
    Wrap data validation logic for a single incoming reading.
    Applies generous physical bounds to numeric sensor columns.
    """
    raw_reading = state.get("raw_reading", {})
    df = pd.DataFrame([raw_reading])

    flagged_df, report_df = validate(df)

    flagged_cols = []
    for c in flagged_df.columns:
        if c.endswith("_flagged") and bool(flagged_df[c].iloc[0]):
            flagged_cols.append(c[:-8])

    is_valid = (len(flagged_cols) == 0)
    report = report_df.to_dict(orient="records") if not report_df.empty else []

    validation_res = {
        "is_valid": is_valid,
        "flagged_columns": flagged_cols,
        "report": report
    }
    return {"validation_result": validation_res}


# --- 3. Node: check_anomaly ---
def check_anomaly(state: PipelineState) -> Dict[str, Any]:
    """
    Loads per-chiller trained physical response model, predicts expected KW,
    computes residual z-score against stored mean/std, and flags |z| > 3.0.
    """
    chiller_id = state.get("chiller_id")
    raw_reading = state.get("raw_reading", {})
    df = pd.DataFrame([raw_reading])

    try:
        agent = get_anomaly_agent(chiller_id)
    except FileNotFoundError:
        return {
            "anomaly_result": {
                "predicted_kw": 0.0,
                "actual_kw": 0.0,
                "z_score": 0.0,
                "is_anomaly": False,
                "error": f"No trained model found for chiller {chiller_id}"
            }
        }

    # Ensure df contains required columns specified in agent's col_map
    for stype, col_name in agent.col_map.items():
        if col_name and col_name not in df.columns:
            matched_col = find_sensor_column(df, stype)
            if matched_col and matched_col in df.columns:
                df[col_name] = df[matched_col]
            else:
                df[col_name] = np.nan

    # Ensure required feature names are present
    for fname in agent.feature_names:
        if fname not in df.columns:
            if fname == "DeltaT":
                inlet = agent.col_map.get("inlet")
                outlet = agent.col_map.get("outlet")
                if inlet and outlet and inlet in df.columns and outlet in df.columns:
                    df["DeltaT"] = df[inlet] - df[outlet]
                else:
                    df["DeltaT"] = np.nan
            elif fname == "Thermal_Load":
                flow = agent.col_map.get("flow")
                if flow and flow in df.columns and "DeltaT" in df.columns:
                    df["Thermal_Load"] = df[flow] * df["DeltaT"]
                else:
                    df["Thermal_Load"] = np.nan
            else:
                df[fname] = np.nan

    detected_df = agent.detect_anomalies(df)

    pred_kw = detected_df["predicted_KW"].iloc[0]
    power_col = agent.col_map.get("power")
    actual_kw = detected_df[power_col].iloc[0] if power_col and power_col in detected_df.columns else np.nan
    z_score = detected_df["z_score"].iloc[0]
    is_anom = bool(detected_df["is_anomalous"].iloc[0])

    anomaly_res = {
        "predicted_kw": float(pred_kw) if pd.notna(pred_kw) else 0.0,
        "actual_kw": float(actual_kw) if pd.notna(actual_kw) else 0.0,
        "z_score": float(z_score) if pd.notna(z_score) else 0.0,
        "is_anomaly": is_anom
    }
    return {"anomaly_result": anomaly_res}


# --- 4. Conditional Edge Router ---
def route_after_anomaly(state: PipelineState) -> str:
    """
    Route based on anomaly result:
    If is_anomaly is True -> generate_insight
    If False -> log_normal
    """
    anomaly_res = state.get("anomaly_result", {})
    if anomaly_res.get("is_anomaly", False):
        return "generate_insight"
    return "log_normal"


# --- 5. Node: generate_insight (Placeholder) ---
def generate_insight(state: PipelineState) -> Dict[str, Any]:
    """Placeholder NLG node for formatting an operational insight string when an anomaly occurs."""
    c_id = state.get("chiller_id")
    ts = state.get("timestamp", "")
    anom_res = state.get("anomaly_result", {})
    act = anom_res.get("actual_kw", 0.0)
    pred = anom_res.get("predicted_kw", 0.0)
    z = anom_res.get("z_score", 0.0)

    alias_info = ""
    if c_id in ASSET_ALIASES and ASSET_ALIASES[c_id]["is_aliased"]:
        canon_id = ASSET_ALIASES[c_id]["canonical_id"]
        alias_info = f" [ALIAS of physical asset {ASSET_ALIASES[c_id]['canonical_name']} (canonical ID {canon_id})]"

    text = (
        f"[ANOMALY DETECTED] Chiller {c_id}{alias_info} at {ts}: "
        f"actual KW {act:.2f} vs expected {pred:.2f}, z-score {z:.2f}"
    )
    return {"insight_text": text}


# --- Node: log_normal ---
def log_normal(state: PipelineState) -> Dict[str, Any]:
    """Node logging normal operational state when no anomaly is detected."""
    c_id = state.get("chiller_id")
    ts = state.get("timestamp", "")
    anom_res = state.get("anomaly_result", {})
    act = anom_res.get("actual_kw", 0.0)
    pred = anom_res.get("predicted_kw", 0.0)
    z = anom_res.get("z_score", 0.0)

    alias_info = ""
    if c_id in ASSET_ALIASES and ASSET_ALIASES[c_id]["is_aliased"]:
        canon_id = ASSET_ALIASES[c_id]["canonical_id"]
        alias_info = f" [ALIAS of physical asset {ASSET_ALIASES[c_id]['canonical_name']} (canonical ID {canon_id})]"

    text = (
        f"[NORMAL] Chiller {c_id}{alias_info} at {ts}: "
        f"actual KW {act:.2f} vs expected {pred:.2f}, z-score {z:.2f}"
    )
    return {"insight_text": text}


# --- Build & Compile LangGraph Pipeline ---
def build_pipeline_graph():
    """Construct and compile the LangGraph workflow graph."""
    builder = StateGraph(PipelineState)

    builder.add_node("validate_reading", validate_reading)
    builder.add_node("check_anomaly", check_anomaly)
    builder.add_node("generate_insight", generate_insight)
    builder.add_node("log_normal", log_normal)

    builder.add_edge(START, "validate_reading")
    builder.add_edge("validate_reading", "check_anomaly")
    builder.add_conditional_edges(
        "check_anomaly",
        route_after_anomaly,
        {
            "generate_insight": "generate_insight",
            "log_normal": "log_normal"
        }
    )
    builder.add_edge("generate_insight", END)
    builder.add_edge("log_normal", END)

    return builder.compile()


# --- 6. Test Harness Data Retrieval ---
def fetch_recent_readings_from_db(chiller_ids: List[int], limit_per_chiller: int = 20) -> List[Dict[str, Any]]:
    """Pull real recent readings for target chillers from PostgreSQL database."""
    try:
        app_conn = psycopg2.connect(
            host=os.environ["APPDB_HOST"],
            port=os.environ["APPDB_PORT"],
            dbname=os.environ["APPDB_NAME"],
            user=os.environ["APPDB_USER"],
            password=os.environ["APPDB_PASSWORD"]
        )
        chiller_tuple = tuple(chiller_ids)
        query_sensors = """
            SELECT me."MachineId" as machine_id, me."Id" as sensor_id, me."SeriesDescription" as series_desc
            FROM "MachineExplorer" me
            JOIN machine m ON m."machineId" = me."MachineId"
            WHERE m."machineId" IN %s AND me."SeriesDescription" IS NOT NULL
        """
        sensors_df = pd.read_sql_query(query_sensors, app_conn, params=(chiller_tuple,))
        app_conn.close()

        if sensors_df.empty:
            return []

        ts_conn = psycopg2.connect(
            host=os.environ["TIMESCALE_HOST"],
            port=os.environ["TIMESCALE_PORT"],
            dbname=os.environ["TIMESCALE_NAME"],
            user=os.environ["TIMESCALE_USER"],
            password=os.environ["TIMESCALE_PASSWORD"]
        )
        sensor_ids = tuple(sensors_df["sensor_id"].tolist())
        query_readings = """
            SELECT ("timestamp" AT TIME ZONE 'Asia/Calcutta') AS timestamp, machineexplorerid, value
            FROM trendseriesmeterdata
            WHERE machineexplorerid IN %s
            ORDER BY timestamp DESC
            LIMIT 5000
        """
        readings_df = pd.read_sql_query(query_readings, ts_conn, params=(sensor_ids,))
        ts_conn.close()

        if readings_df.empty:
            return []

        merged = readings_df.merge(sensors_df, left_on="machineexplorerid", right_on="sensor_id")
        merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.round("15min")

        results = []
        for c_id in chiller_ids:
            c_df = merged[merged["machine_id"] == c_id]
            if c_df.empty:
                continue
            timestamps = c_df["timestamp"].drop_duplicates().head(limit_per_chiller).tolist()
            for ts in timestamps:
                ts_df = c_df[c_df["timestamp"] == ts]
                raw_reading = ts_df.set_index("series_desc")["value"].to_dict()
                raw_reading["machineId"] = c_id
                raw_reading["timestamp"] = str(ts)
                results.append({
                    "chiller_id": c_id,
                    "timestamp": str(ts),
                    "raw_reading": raw_reading
                })
        return results

    except Exception as e:
        print(f"[Warning] Failed to fetch readings from DB: {e}. Falling back to trend_wide.csv")
        return fetch_recent_readings_from_csv(chiller_ids, limit_per_chiller)


def fetch_recent_readings_from_csv(chiller_ids: List[int], limit_per_chiller: int = 20) -> List[Dict[str, Any]]:
    """Fallback reader from data/trend_wide.csv."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "trend_wide.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    results = []
    for c_id in chiller_ids:
        c_df = df[df["machineId"] == c_id].dropna(how="all").tail(limit_per_chiller)
        for _, row in c_df.iterrows():
            reading_dict = row.dropna().to_dict()
            ts = str(reading_dict.get("timestamp", ""))
            results.append({
                "chiller_id": c_id,
                "timestamp": ts,
                "raw_reading": reading_dict
            })
    return results


def run_test_harness(chiller_ids: List[int] = [4054, 2828, 2821], limit_per_chiller: int = 20):
    """
    Test Harness: Pulls 20 recent real readings for target chillers from DB,
    runs each through the LangGraph pipeline, and prints the state at each step.
    """
    print("=" * 100)
    print("LANGGRAPH PIPELINE TEST HARNESS — END-TO-END EXECUTION")
    print(f"Target Chillers: {chiller_ids} ({limit_per_chiller} readings each)")
    print("=" * 100)

    app = build_pipeline_graph()

    readings = fetch_recent_readings_from_db(chiller_ids, limit_per_chiller=limit_per_chiller)
    print(f"Total test readings fetched: {len(readings)}\n")

    summary_counts = {"total": len(readings), "valid": 0, "anomalies": 0, "normal": 0}

    for idx, item in enumerate(readings, 1):
        c_id = item["chiller_id"]
        ts = item["timestamp"]
        initial_state: PipelineState = {
            "chiller_id": c_id,
            "timestamp": ts,
            "raw_reading": item["raw_reading"],
            "validation_result": {},
            "anomaly_result": {},
            "insight_text": None
        }

        final_state = app.invoke(initial_state)

        is_valid = final_state["validation_result"].get("is_valid", True)
        anom_res = final_state["anomaly_result"]
        is_anom = anom_res.get("is_anomaly", False)

        if is_valid:
            summary_counts["valid"] += 1
        if is_anom:
            summary_counts["anomalies"] += 1
        else:
            summary_counts["normal"] += 1

        alias_str = ""
        if c_id in ASSET_ALIASES and ASSET_ALIASES[c_id]["is_aliased"]:
            canon_id = ASSET_ALIASES[c_id]["canonical_id"]
            alias_str = f" [ALIAS -> {ASSET_ALIASES[c_id]['canonical_name']} (ID {canon_id})]"

        print(f"[{idx:02d}/{len(readings):02d}] Chiller {c_id}{alias_str} | Timestamp: {ts}")
        print(f"     Validation : Valid={is_valid} | Flagged Cols={final_state['validation_result'].get('flagged_columns', [])}")
        print(f"     Anomaly    : Actual KW={anom_res.get('actual_kw', 0.0):.2f} | Expected={anom_res.get('predicted_kw', 0.0):.2f} | Z-Score={anom_res.get('z_score', 0.0):.2f} | IsAnomaly={is_anom}")
        print(f"     Insight    : {final_state.get('insight_text')}")
        print("-" * 100)

    print("\n" + "=" * 100)
    print("PIPELINE TEST HARNESS SUMMARY")
    print(f"Total Readings Processed: {summary_counts['total']}")
    print(f"Sanity Valid Readings   : {summary_counts['valid']} / {summary_counts['total']}")
    print(f"Anomalies Flagged (|z|>3): {summary_counts['anomalies']}")
    print(f"Normal Operation Logged : {summary_counts['normal']}")
    print("=" * 100)


if __name__ == "__main__":
    run_test_harness()
