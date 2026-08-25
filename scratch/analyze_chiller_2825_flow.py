import os
import sys
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.anomaly_agent import AnomalyAgent, MODEL_SAVE_DIR

load_dotenv()

DB_TIMEZONE = "Asia/Calcutta"

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

agent_2825 = AnomalyAgent.load(2825, save_dir=MODEL_SAVE_DIR)
power_col = agent_2825.col_map["power"]
flow_col = agent_2825.col_map["flow"]
inlet_col = agent_2825.col_map["inlet"]
outlet_col = agent_2825.col_map["outlet"]

# Fill power and flow from synonymous columns if primary is NaN
if "CH-1, POWER CONSUMPTION (KW)" in wide_df.columns:
    wide_df[power_col] = wide_df[power_col].fillna(wide_df["CH-1, POWER CONSUMPTION (KW)"])
if "CHW FLOW RATE (m3/h)" in wide_df.columns:
    wide_df[flow_col] = wide_df[flow_col].fillna(wide_df["CHW FLOW RATE (m3/h)"])

wide_df["DeltaT"] = wide_df[inlet_col] - wide_df[outlet_col]

wide_df["date_str"] = wide_df["timestamp"].dt.strftime("%Y-%m-%d")
hist_clean_df = wide_df[wide_df["date_str"] != "2026-08-07"].copy()
aug7_df = wide_df[wide_df["date_str"] == "2026-08-07"].copy()

print("=" * 100)
print("CHILLER 2825: FLOW & DELTAT DISTRIBUTION COMPARISON AT HIGH POWER (800+ KW)")
print(f"Power Column: '{power_col}' | Flow Column: '{flow_col}'")
print("=" * 100)

high_power_hist = hist_clean_df[hist_clean_df[power_col] >= 800.0].copy()

print(f"\n1. HISTORICAL FLOW DISTRIBUTION AT HIGH POWER (Power >= 800 kW: {len(high_power_hist)} samples):")
f_hp = high_power_hist[flow_col].dropna()
print(f"   Mean Flow        : {f_hp.mean():.2f} m3/h")
print(f"   Std Dev Flow     : {f_hp.std():.2f} m3/h")
print(f"   Min Flow         : {f_hp.min():.2f} m3/h")
print(f"   25th Percentile  : {f_hp.quantile(0.25):.2f} m3/h")
print(f"   Median Flow      : {f_hp.quantile(0.50):.2f} m3/h")
print(f"   75th Percentile  : {f_hp.quantile(0.75):.2f} m3/h")
print(f"   Max Flow         : {f_hp.max():.2f} m3/h")

print(f"\n2. HISTORICAL DELTAT DISTRIBUTION AT HIGH POWER (Power >= 800 kW):")
dt_hp = high_power_hist["DeltaT"].dropna()
print(f"   Mean DeltaT      : {dt_hp.mean():.2f} °C")
print(f"   Std Dev DeltaT   : {dt_hp.std():.2f} °C")
print(f"   Min DeltaT       : {dt_hp.min():.2f} °C")
print(f"   Median DeltaT    : {dt_hp.quantile(0.50):.2f} °C")
print(f"   Max DeltaT       : {dt_hp.max():.2f} °C")

print(f"\n3. AUGUST 7, 2026 EVENT (Power = 880-887 kW: {len(aug7_df)} samples):")
f_aug7 = aug7_df[flow_col].dropna()
dt_aug7 = aug7_df["DeltaT"].dropna()
print(f"   Aug 7 Mean Flow     : {f_aug7.mean():.2f} m3/h  (vs Normal High-Power Mean: {f_hp.mean():.2f} m3/h)")
print(f"   Aug 7 Min-Max Flow  : {f_aug7.min():.2f} - {f_aug7.max():.2f} m3/h")
print(f"   Aug 7 Mean DeltaT   : {dt_aug7.mean():.2f} °C   (vs Normal High-Power Mean: {dt_hp.mean():.2f} °C)")
print(f"   Aug 7 Min-Max DeltaT: {dt_aug7.min():.2f} - {dt_aug7.max():.2f} °C")

# Quantify Flow Z-score on Aug 7 against Historical High-Power Flow distribution
flow_mean_hp = f_hp.mean()
flow_std_hp = f_hp.std()
flow_z_aug7 = (f_aug7.mean() - flow_mean_hp) / flow_std_hp if flow_std_hp > 0 else np.nan

print("\n" * 2 + "=" * 100)
print("INDEPENDENT FLOW ANOMALY EVALUATION:")
print("=" * 100)
print(f"Normal Flow at High Power (>= 800 kW) : {flow_mean_hp:.2f} ± {flow_std_hp:.2f} m3/h")
print(f"Aug 7 Observed Flow at High Power     : {f_aug7.mean():.2f} m3/h")
print(f"Flow Deficit / Reduction              : {flow_mean_hp - f_aug7.mean():.2f} m3/h ({(1 - f_aug7.mean()/flow_mean_hp)*100:.1f}% Flow Reduction!)")
print(f"Flow Deficit Z-Score (vs 800+ kW norm): {flow_z_aug7:.2f} sigma")

dt_mean_hp = dt_hp.mean()
dt_std_hp = dt_hp.std()
dt_z_aug7 = (dt_aug7.mean() - dt_mean_hp) / dt_std_hp if dt_std_hp > 0 else np.nan

print(f"\nNormal DeltaT at High Power (>= 800 kW): {dt_mean_hp:.2f} ± {dt_std_hp:.2f} °C")
print(f"Aug 7 Observed DeltaT                 : {dt_aug7.mean():.2f} °C")
print(f"DeltaT Elevation Z-Score              : {dt_z_aug7:+.2f} sigma")
