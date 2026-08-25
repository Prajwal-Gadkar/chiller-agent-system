import os
import sys
import warnings
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.pipeline import build_pipeline_graph, PipelineState, MODEL_SAVE_DIR
from agents.anomaly_agent import AnomalyAgent

warnings.filterwarnings("ignore")
load_dotenv()

DB_TIMEZONE = "Asia/Calcutta"

model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith(".pkl")]
model_chiller_ids = sorted([int(f.split("_")[1]) for f in model_files])

print(f"Auditing consecutive Z > 2.0 sequences across {len(model_chiller_ids)} chillers...\n")

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

query_meta = """
    SELECT m."machineId", me."Id" AS "MachineExplorerId", me."SeriesDescription"
    FROM machine m
    JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
    WHERE m."machineId" IN %(m_ids)s AND me."SeriesDescription" IS NOT NULL;
"""
meta_df = pd.read_sql_query(query_meta, app_conn, params={"m_ids": tuple(model_chiller_ids)})
app_conn.close()

pipeline_app = build_pipeline_graph()

consecutive_events = []

for c_id in model_chiller_ids:
    m_sensors = meta_df[meta_df["machineId"] == c_id]
    if m_sensors.empty:
        continue
    sensor_ids = tuple([int(sid) for sid in m_sensors["MachineExplorerId"].unique().tolist()])
    
    kw_sensor = m_sensors[m_sensors["SeriesDescription"].str.contains("KW|power|temp|flow", case=False, na=False)]
    target_sensor_id = int(kw_sensor["MachineExplorerId"].iloc[0]) if not kw_sensor.empty else sensor_ids[0]
    
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
    
    chiller_z_series = []
    
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
        
        chiller_z_series.append({
            "timestamp": ts,
            "z_score": z,
            "abs_z": abs(z) if pd.notna(z) else 0.0,
            "actual_kw": anom_res.get("actual_kw", 0.0),
            "predicted_kw": anom_res.get("predicted_kw", 0.0)
        })
        
    df_chiller = pd.DataFrame(chiller_z_series).sort_values("timestamp").reset_index(drop=True)
    if df_chiller.empty:
        continue
        
    # Scan for consecutive runs of abs_z > 2.0
    df_chiller["is_above_2"] = df_chiller["abs_z"] > 2.0
    
    # Identify contiguous blocks
    df_chiller["block_id"] = (df_chiller["is_above_2"] != df_chiller["is_above_2"].shift()).cumsum()
    
    for block_id, group in df_chiller.groupby("block_id"):
        if group["is_above_2"].iloc[0] and len(group) >= 4:
            consecutive_events.append({
                "chiller_id": c_id,
                "start_time": str(group["timestamp"].iloc[0]),
                "end_time": str(group["timestamp"].iloc[-1]),
                "consecutive_count": len(group),
                "peak_abs_z": group["abs_z"].max(),
                "mean_z": group["z_score"].mean(),
                "mean_actual": group["actual_kw"].mean(),
                "mean_predicted": group["predicted_kw"].mean(),
                "mean_residual": group["actual_kw"].mean() - group["predicted_kw"].mean()
            })

ts_conn.close()

print("=" * 100)
print("FLEET SCAN FOR >= 4 CONSECUTIVE READINGS WITH |Z| > 2.0")
print("=" * 100)
print(f"Total Chillers Audited: {len(model_chiller_ids)}")
print(f"Chillers Triggering Sustained Sequence Rule (>= 4 consecutive |z| > 2.0): {len(consecutive_events)}\n")

if consecutive_events:
    df_events = pd.DataFrame(consecutive_events)
    print(df_events[["chiller_id", "start_time", "end_time", "consecutive_count", "peak_abs_z", "mean_z", "mean_actual", "mean_predicted", "mean_residual"]].to_string(index=False))
else:
    print("No chillers triggered the sustained sequence rule.")
