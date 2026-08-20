import os
import sys
import warnings
import pandas as pd
import psycopg2
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.anomaly_agent import AnomalyAgent, MODEL_SAVE_DIR

warnings.filterwarnings("ignore")
load_dotenv()

DB_TIMEZONE = "Asia/Calcutta"
TRAIN_WINDOW_START = "2026-05-01"

chiller_ids = [3392, 3894, 4054]

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
    WHERE m."machineId" IN %(m_ids)s;
"""
meta_df = pd.read_sql_query(query_meta, app_conn, params={"m_ids": tuple(chiller_ids)})
app_conn.close()

print(f"Refitting models for Chillers {chiller_ids} using corrected KW ValueY sensor...\n")

for m_id in chiller_ids:
    m_sensors = meta_df[meta_df["machineId"] == m_id].dropna(subset=["SeriesDescription"])
    sensor_ids = tuple(m_sensors["MachineExplorerId"].unique().tolist())
    
    query_ts = """
        SELECT ("timestamp" AT TIME ZONE %(tz)s) AS "timestamp", machineexplorerid, value
        FROM trendseriesmeterdata
        WHERE machineexplorerid IN %(sensor_ids)s
          AND "timestamp" >= %(start_date)s
    """
    params = {"sensor_ids": sensor_ids, "tz": DB_TIMEZONE, "start_date": TRAIN_WINDOW_START}
    raw_df = pd.read_sql_query(query_ts, ts_conn, params=params)
    
    raw_df = raw_df.merge(m_sensors[["MachineExplorerId", "SeriesDescription"]], left_on="machineexplorerid", right_on="MachineExplorerId")
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"]).dt.round("15min")

    wide_df = raw_df.pivot_table(
        index="timestamp",
        columns="SeriesDescription",
        values="value",
        aggfunc="first"
    ).reset_index()

    agent = AnomalyAgent(machine_id=m_id)
    cv_metrics = agent.fit(wide_df, regime_start=TRAIN_WINDOW_START)
    save_path = agent.save(save_dir=MODEL_SAVE_DIR)
    
    print(f"Chiller {m_id:4d} | Power Col: {agent.col_map['power']} | R2 CV: {cv_metrics['R2']:.4f} | RMSE: {cv_metrics['RMSE']:.2f} kW | Residual Mean/Std: {agent.residual_mean:.2f}/{agent.residual_std:.2f} | Saved: {save_path}")

ts_conn.close()
print("\nRefitting complete!")
