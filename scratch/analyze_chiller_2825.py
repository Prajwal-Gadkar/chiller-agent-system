import os
import sys
import pickle
import warnings
import pandas as pd
import numpy as np
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

# 1. Load Chiller 2825 model and metadata
agent_2825 = AnomalyAgent.load(2825, save_dir=MODEL_SAVE_DIR)
print("=" * 100)
print("CHILLER 2825 MODEL METADATA & RESIDUAL STATISTICS")
print("=" * 100)
print(f"Machine ID      : {agent_2825.machine_id}")
print(f"Power Column    : {agent_2825.col_map.get('power')}")
print(f"Flow Column     : {agent_2825.col_map.get('flow')}")
print(f"Inlet Column    : {agent_2825.col_map.get('inlet')}")
print(f"Outlet Column   : {agent_2825.col_map.get('outlet')}")
print(f"Features        : {agent_2825.feature_names}")
print(f"Residual Mean   : {agent_2825.residual_mean:.4f} kW")
print(f"Residual Std    : {agent_2825.residual_std:.4f} kW")
print(f"CV R2 Score     : {agent_2825.cv_metrics.get('R2'):.4f}")
print(f"CV RMSE         : {agent_2825.cv_metrics.get('RMSE'):.2f} kW")
print(f"Training Samples: {agent_2825.cv_metrics.get('n_samples')}")

# 2. Pull Chiller 2825 data from PostgreSQL DB
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
    WHERE m."machineId" = 2825 AND me."SeriesDescription" IS NOT NULL;
"""
m_sensors = pd.read_sql_query(query_meta, app_conn)
app_conn.close()

sensor_ids = tuple([int(sid) for sid in m_sensors["MachineExplorerId"].unique().tolist()])

# Pull data for Chiller 2825 post-2026-05-01
query_ts = """
    SELECT ("timestamp" AT TIME ZONE %(tz)s) AS "timestamp", machineexplorerid, value
    FROM trendseriesmeterdata
    WHERE machineexplorerid IN %(sensor_ids)s
      AND "timestamp" >= '2026-05-01'
"""
raw_df = pd.read_sql_query(query_ts, ts_conn, params={"sensor_ids": sensor_ids, "tz": DB_TIMEZONE})
ts_conn.close()

raw_df = raw_df.merge(m_sensors[["MachineExplorerId", "SeriesDescription"]], left_on="machineexplorerid", right_on="MachineExplorerId")
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"]).dt.round("15min")

wide_df = raw_df.pivot_table(
    index="timestamp",
    columns="SeriesDescription",
    values="value",
    aggfunc="first"
).reset_index()

power_col = agent_2825.col_map["power"]
flow_col = agent_2825.col_map["flow"]
inlet_col = agent_2825.col_map["inlet"]
outlet_col = agent_2825.col_map["outlet"]

print("\n" * 2 + "=" * 100)
print(f"HISTORICAL POWER METRICS FOR CHILLER 2825 (POST 2026-05-01: {len(wide_df)} ROWS)")
print("=" * 100)
p_series = wide_df[power_col].dropna()
print(f"Mean Power      : {p_series.mean():.2f} kW")
print(f"Std Dev Power   : {p_series.std():.2f} kW")
print(f"Min Power       : {p_series.min():.2f} kW")
print(f"25th Percentile : {p_series.quantile(0.25):.2f} kW")
print(f"Median Power    : {p_series.quantile(0.50):.2f} kW")
print(f"75th Percentile : {p_series.quantile(0.75):.2f} kW")
print(f"90th Percentile : {p_series.quantile(0.90):.2f} kW")
print(f"95th Percentile : {p_series.quantile(0.95):.2f} kW")
print(f"99th Percentile : {p_series.quantile(0.99):.2f} kW")
print(f"Max Power       : {p_series.max():.2f} kW")

# Evaluate response model & anomalies on full dataset
detected_df = agent_2825.detect_anomalies(wide_df)

# Focus on 2026-08-07
wide_df["date"] = wide_df["timestamp"].dt.date
aug7_df = detected_df[detected_df["timestamp"].dt.strftime("%Y-%m-%d") == "2026-08-07"].copy()

print("\n" * 2 + "=" * 100)
print(f"2026-08-07 FULL DAY PROFILE FOR CHILLER 2825 ({len(aug7_df)} READINGS)")
print("=" * 100)

cols_to_show = ["timestamp", power_col, flow_col, inlet_col, outlet_col, "predicted_KW", "residual_KW", "z_score", "is_anomalous"]
aug7_sub = aug7_df[cols_to_show].sort_values("timestamp")

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)
print(aug7_sub.to_string(index=False))

print("\n" * 2 + "=" * 100)
print("AUG 7 ANOMALY SUMMARY FOR CHILLER 2825:")
print("=" * 100)
print(f"Aug 7 Mean Actual KW   : {aug7_df[power_col].mean():.2f} kW")
print(f"Aug 7 Max Actual KW    : {aug7_df[power_col].max():.2f} kW")
print(f"Aug 7 Mean Predicted KW: {aug7_df['predicted_KW'].mean():.2f} kW")
print(f"Aug 7 Max Residual KW  : {aug7_df['residual_KW'].max():.2f} kW")
print(f"Aug 7 Max Z-Score      : {aug7_df['z_score'].max():.2f}")
print(f"Aug 7 Count |z| > 2.0  : {(aug7_df['z_score'].abs() > 2.0).sum()} / {len(aug7_df)}")
print(f"Aug 7 Count |z| > 2.5  : {(aug7_df['z_score'].abs() > 2.5).sum()} / {len(aug7_df)}")
print(f"Aug 7 Count |z| > 3.0  : {(aug7_df['z_score'].abs() > 3.0).sum()} / {len(aug7_df)}")
