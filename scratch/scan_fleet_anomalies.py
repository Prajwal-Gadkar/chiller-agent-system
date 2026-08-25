import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.pipeline import build_pipeline_graph, PipelineState, get_anomaly_agent, find_sensor_col_single_row, MODEL_SAVE_DIR
from agents.anomaly_agent import AnomalyAgent

warnings.filterwarnings("ignore")
load_dotenv()

DB_TIMEZONE = "Asia/Calcutta"

# Find all chillers with trained models
model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith(".pkl")]
model_chiller_ids = sorted([int(f.split("_")[1]) for f in model_files])

print(f"Scanning recent readings across {len(model_chiller_ids)} chillers with trained models...\n")

app_conn = psycopg2.connect(
    host=os.environ["APPDB_HOST"],
    port=os.environ["APPDB_PORT"],
    dbname=os.environ["APPDB_NAME"],
    user=os.environ["APPDB_USER"],
    password=os.environ["APPDB_PASSWORD"]
)

ts_conn = psycopg2.connect(
    host=os.environ["TIMESCALE_HOST"],
    port=os.environ["TIMESCALE_PORT"],
    dbname=os.environ["TIMESCALE_NAME"],
    user=os.environ["TIMESCALE_USER"],
    password=os.environ["TIMESCALE_PASSWORD"]
)

all_z_scores = []
anomaly_hits = []

# Fetch sensor metadata for all candidate chillers
query_meta = """
    SELECT m."machineId", me."Id" AS "MachineExplorerId", me."SeriesDescription"
    FROM machine m
    JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
    WHERE m."machineId" IN %(m_ids)s AND me."SeriesDescription" IS NOT NULL;
"""
meta_df = pd.read_sql_query(query_meta, app_conn, params={"m_ids": tuple(model_chiller_ids)})
app_conn.close()

pipeline_app = build_pipeline_graph()

processed_readings = []

for c_id in model_chiller_ids:
    m_sensors = meta_df[meta_df["machineId"] == c_id]
    if m_sensors.empty:
        continue
    sensor_ids = tuple([int(sid) for sid in m_sensors["MachineExplorerId"].unique().tolist()])
    
    # Find telemetry sensor (kw / power / flow / temp)
    kw_sensor = m_sensors[m_sensors["SeriesDescription"].str.contains("KW|power|temp|flow", case=False, na=False)]
    target_sensor_id = int(kw_sensor["MachineExplorerId"].iloc[0]) if not kw_sensor.empty else sensor_ids[0]
    
    # Query 50 recent timestamps
    query_ts = """
        SELECT ("timestamp" AT TIME ZONE %(tz)s) AS timestamp
        FROM trendseriesmeterdata
        WHERE machineexplorerid = %(target_id)s
        ORDER BY timestamp DESC
        LIMIT 50
    """
    ts_df = pd.read_sql_query(query_ts, ts_conn, params={"target_id": target_sensor_id, "tz": DB_TIMEZONE})
    if ts_df.empty:
        continue
    
    timestamps = tuple(ts_df["timestamp"].tolist())
    
    query_readings = """
        SELECT ("timestamp" AT TIME ZONE %(tz)s) AS timestamp, machineexplorerid, value
        FROM trendseriesmeterdata
        WHERE machineexplorerid IN %(sensor_ids)s AND ("timestamp" AT TIME ZONE %(tz)s) IN %(timestamps)s
    """
    readings_df = pd.read_sql_query(query_readings, ts_conn, params={"sensor_ids": sensor_ids, "timestamps": timestamps, "tz": DB_TIMEZONE})
    if readings_df.empty:
        continue

    merged = readings_df.merge(m_sensors[["MachineExplorerId", "SeriesDescription"]], left_on="machineexplorerid", right_on="MachineExplorerId")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.round("15min")
    
    distinct_ts = sorted(merged["timestamp"].drop_duplicates().tolist())
    
    for ts in distinct_ts:
        ts_sub = merged[merged["timestamp"] == ts]
        raw_reading = ts_sub.set_index("SeriesDescription")["value"].to_dict()
        raw_reading["machineId"] = c_id
        raw_reading["timestamp"] = str(ts)
        
        initial_state: PipelineState = {
            "chiller_id": c_id,
            "timestamp": str(ts),
            "raw_reading": raw_reading,
            "validation_result": {},
            "anomaly_result": {},
            "insight_text": None
        }
        
        final_state = pipeline_app.invoke(initial_state)
        anom_res = final_state.get("anomaly_result", {})
        z = anom_res.get("z_score", 0.0)
        
        if pd.notna(z) and abs(z) > 0.0001:
            all_z_scores.append({
                "chiller_id": c_id,
                "timestamp": str(ts),
                "z_score": z,
                "abs_z": abs(z),
                "actual_kw": anom_res.get("actual_kw"),
                "predicted_kw": anom_res.get("predicted_kw"),
                "state": final_state
            })
            
            if anom_res.get("is_anomaly", False):
                anomaly_hits.append(final_state)

ts_conn.close()

z_df = pd.DataFrame(all_z_scores)

print("=" * 100)
print(f"FLEET ANOMALY SCAN RESULTS — {len(model_chiller_ids)} CHILLERS AUDITED")
print("=" * 100)
print(f"Total Non-Zero Z-Scores Calculated: {len(z_df)}")

if not z_df.empty:
    print(f"Min Z-Score        : {z_df['z_score'].min():.4f}")
    print(f"Max Z-Score        : {z_df['z_score'].max():.4f}")
    print(f"Max Absolute Z     : {z_df['abs_z'].max():.4f}")
    print(f"Count |z| > 1.5    : {(z_df['abs_z'] > 1.5).sum()}")
    print(f"Count |z| > 2.0    : {(z_df['abs_z'] > 2.0).sum()}")
    print(f"Count |z| > 2.5    : {(z_df['abs_z'] > 2.5).sum()}")
    print(f"Count |z| > 3.0    : {(z_df['abs_z'] > 3.0).sum()}")

print("\n" + "=" * 100)
print(f"GENUINE ANOMALY HITS FOUND (|z| > 3.0): {len(anomaly_hits)}")
print("=" * 100)

if anomaly_hits:
    print("\nFirst Genuine Anomaly Hit Pipeline Output:")
    hit = anomaly_hits[0]
    print(f"Chiller ID   : {hit['chiller_id']}")
    print(f"Timestamp    : {hit['timestamp']}")
    print(f"Actual KW    : {hit['anomaly_result'].get('actual_kw'):.2f} kW")
    print(f"Predicted KW : {hit['anomaly_result'].get('predicted_kw'):.2f} kW")
    print(f"Z-Score      : {hit['anomaly_result'].get('z_score'):.2f}")
    print(f"Insight Text : {hit['insight_text']}")
    print(f"\nFull State Object:\n")
    print(json.dumps(hit, indent=2))
else:
    print("\nTop 5 Highest |z-score| Readings Across Fleet:")
    top_z = z_df.sort_values("abs_z", ascending=False).head(5)
    for idx, row in top_z.iterrows():
        print(f"Chiller {row['chiller_id']:4d} at {row['timestamp']} | Actual: {row['actual_kw']:.2f} kW | Pred: {row['predicted_kw']:.2f} kW | Z: {row['z_score']:+.2f}")
