import os
import sys
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

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

power_col = "CH-1, POWER CONSUMPTION (KW)"
flow_col = "CHW FLOW RATE (m3/h)"
inlet_col = "Evaporator_Inlet_Temp"
outlet_col = "Evaporator_Outlet_Temp"

wide_df["DeltaT"] = wide_df[inlet_col] - wide_df[outlet_col]
wide_df["Thermal_Load"] = wide_df[flow_col] * wide_df["DeltaT"]

wide_df["date_str"] = wide_df["timestamp"].dt.strftime("%Y-%m-%d")
hist_clean_df = wide_df[wide_df["date_str"] != "2026-08-07"].copy()
aug7_df = wide_df[wide_df["date_str"] == "2026-08-07"].copy()

high_power_hist = hist_clean_df[hist_clean_df[power_col] >= 800.0].copy()

print("=" * 100)
print("CHILLER 2825 TRUE PHYSICAL ANALYSIS: HIGH POWER (800+ KW) NORMAL VS AUG 7 EVENT")
print("=" * 100)

print(f"\n1. HISTORICAL NORMAL AT HIGH POWER (Power >= 800 kW: {len(high_power_hist)} 15-min samples):")
f_hp = high_power_hist[flow_col].dropna()
dt_hp = high_power_hist["DeltaT"].dropna()
tl_hp = high_power_hist["Thermal_Load"].dropna()

print(f"   Flow (m3/h)        : Mean {f_hp.mean():.2f} | Std {f_hp.std():.2f} | Range [{f_hp.min():.2f}, {f_hp.max():.2f}] | Median {f_hp.median():.2f}")
print(f"   DeltaT (deg C)     : Mean {dt_hp.mean():.2f} | Std {dt_hp.std():.2f} | Range [{dt_hp.min():.2f}, {dt_hp.max():.2f}] | Median {dt_hp.median():.2f}")
print(f"   Thermal Load (L*dT): Mean {tl_hp.mean():.2f} | Std {tl_hp.std():.2f} | Range [{tl_hp.min():.2f}, {tl_hp.max():.2f}] | Median {tl_hp.median():.2f}")

print(f"\n2. AUGUST 7, 2026 EVENT (Power = 880-887 kW: {len(aug7_df)} samples):")
f_aug = aug7_df[flow_col].dropna()
dt_aug = aug7_df["DeltaT"].dropna()
tl_aug = aug7_df["Thermal_Load"].dropna()

print(f"   Aug 7 Flow (m3/h)  : Mean {f_aug.mean():.2f} | Range [{f_aug.min():.2f}, {f_aug.max():.2f}]")
print(f"   Aug 7 DeltaT (C)   : Mean {dt_aug.mean():.2f} | Range [{dt_aug.min():.2f}, {dt_aug.max():.2f}]")
print(f"   Aug 7 Thermal Load : Mean {tl_aug.mean():.2f} | Range [{tl_aug.min():.2f}, {tl_aug.max():.2f}]")

print("\n" * 2 + "=" * 100)
print("PHYSICAL COMPARISON & ANOMALY SIGNALS:")
print("=" * 100)

flow_diff = f_hp.mean() - f_aug.mean()
flow_z = (f_aug.mean() - f_hp.mean()) / f_hp.std()

dt_diff = dt_hp.mean() - dt_aug.mean()
dt_z = (dt_aug.mean() - dt_hp.mean()) / dt_hp.std()

tl_diff = tl_hp.mean() - tl_aug.mean()
tl_z = (tl_aug.mean() - tl_hp.mean()) / tl_hp.std()

print(f"Flow Deficit      : Normal 800+ kW Flow = {f_hp.mean():.2f} m3/h vs Aug 7 Flow = {f_aug.mean():.2f} m3/h (Diff: {flow_diff:+.2f} m3/h, Z: {flow_z:+.2f} sigma)")
print(f"DeltaT Elevation  : Normal 800+ kW DeltaT = {dt_hp.mean():.2f} °C vs Aug 7 DeltaT = {dt_aug.mean():.2f} °C (Diff: {-dt_diff:+.2f} °C, Z: {dt_z:+.2f} sigma)")
print(f"Thermal Load      : Normal 800+ kW Thermal Load = {tl_hp.mean():.2f} vs Aug 7 Thermal Load = {tl_aug.mean():.2f} (Diff: {tl_diff:+.2f}, Z: {tl_z:+.2f} sigma)")
