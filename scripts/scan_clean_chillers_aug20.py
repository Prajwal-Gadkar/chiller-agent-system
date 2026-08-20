"""
Scan Clean Chillers (Aug 2026 Fleet Update - Batch Optimized)

Scans all chiller assets in Persistent_AppDb_Aug20 and checks data corruption rates
on the universal 5-column set (KW, Flow, inlet_temperature, outlet_temperature, DeltaT)
across the TimescaleDB dataset.

Saves results to data/clean_chillers_aug20.csv.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate
from dotenv import load_dotenv

load_dotenv()

DB_TIMEZONE = "Asia/Calcutta"


def get_appdb_connection():
    return psycopg2.connect(
        host=os.environ["APPDB_HOST"],
        port=os.environ["APPDB_PORT"],
        dbname=os.environ["APPDB_NAME"],
        user=os.environ["APPDB_USER"],
        password=os.environ["APPDB_PASSWORD"],
    )


def get_timescale_connection():
    return psycopg2.connect(
        host=os.environ["TIMESCALE_HOST"],
        port=os.environ["TIMESCALE_PORT"],
        dbname=os.environ["TIMESCALE_NAME"],
        user=os.environ["TIMESCALE_USER"],
        password=os.environ["TIMESCALE_PASSWORD"],
    )


def get_chiller_sensor_mapping():
    conn = get_appdb_connection()
    query = """
        SELECT
            m."machineId",
            m."status",
            m."Criticality",
            me."Id" AS "MachineExplorerId",
            me."SeriesDescription"
        FROM machine m
        JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
        WHERE m."machineType" = 'Chiller'
        ORDER BY m."machineId", me."Id";
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def find_column(cols, keywords, exclude=None):
    for c in cols:
        c_lower = c.lower()
        if exclude and any(ex.lower() in c_lower for ex in exclude):
            continue
        if any(kw.lower() in c_lower for kw in keywords):
            return c
    return None


def scan_chillers(start_date="2026-06-01", end_date="2026-08-20"):
    chiller_map = get_chiller_sensor_mapping()
    all_sensor_df = chiller_map.dropna(subset=["SeriesDescription"])
    chillers = sorted(all_sensor_df["machineId"].unique())
    print(f"Loaded {len(chillers)} distinct chiller assets with configured sensors from AppDb.", flush=True)

    conn_ts = get_timescale_connection()

    records = []

    # Batch query in chunks of 150 sensors
    sensor_id_list = all_sensor_df["MachineExplorerId"].unique().tolist()
    chunk_size = 150

    print(f"Pulling sensor data from TimescaleDB for {len(sensor_id_list)} sensors ({start_date} to {end_date})...", flush=True)
    raw_chunks = []
    for i in range(0, len(sensor_id_list), chunk_size):
        chunk_ids = tuple(sensor_id_list[i : i + chunk_size])
        query_data = """
            SELECT ("timestamp" AT TIME ZONE %(tz)s) AS "timestamp", machineexplorerid, value
            FROM trendseriesmeterdata
            WHERE machineexplorerid IN %(sensor_ids)s
              AND "timestamp" >= %(start)s
              AND "timestamp" < %(end)s
        """
        params = {
            "sensor_ids": chunk_ids,
            "start": start_date,
            "end": end_date,
            "tz": DB_TIMEZONE
        }
        chunk_df = pd.read_sql_query(query_data, conn_ts, params=params)
        raw_chunks.append(chunk_df)
        print(f"Loaded chunk {i//chunk_size + 1}/{(len(sensor_id_list) + chunk_size - 1)//chunk_size} ({len(chunk_df)} rows)", flush=True)


    conn_ts.close()

    if not raw_chunks or all(c.empty for c in raw_chunks):
        print("No sensor data returned from TimescaleDB.", flush=True)
        return pd.DataFrame()

    full_raw_df = pd.concat(raw_chunks, ignore_index=True)
    print(f"Loaded {len(full_raw_df)} total sensor readings. Processing per chiller...", flush=True)

    # Merge metadata
    full_raw_df = full_raw_df.merge(all_sensor_df[["MachineExplorerId", "machineId", "SeriesDescription"]], left_on="machineexplorerid", right_on="MachineExplorerId")
    full_raw_df["timestamp"] = pd.to_datetime(full_raw_df["timestamp"]).dt.round("15min")

    for m_id in chillers:
        m_df = full_raw_df[full_raw_df["machineId"] == m_id]
        if m_df.empty:
            records.append({
                "machineId": m_id,
                "n_rows": 0,
                "has_kw": False,
                "has_flow": False,
                "has_inlet": False,
                "has_outlet": False,
                "kw_flag_pct": np.nan,
                "flow_flag_pct": np.nan,
                "inlet_flag_pct": np.nan,
                "outlet_flag_pct": np.nan,
                "status": "NO_DATA_IN_WINDOW"
            })
            print(f"Chiller {m_id:4d} | Rows:     0 | Status: NO_DATA_IN_WINDOW", flush=True)
            continue

        wide_df = m_df.pivot_table(
            index="timestamp",
            columns="SeriesDescription",
            values="value",
            aggfunc="first"
        ).reset_index()

        cols = wide_df.columns.tolist()

        flow_col = find_column(cols, ["flow"], exclude=["condenser"])
        kw_col = find_column(cols, ["kw", "power"], exclude=["ikw", "kwh"])
        inlet_col = find_column(cols, ["inlet", "return"], exclude=["condenser"])
        outlet_col = find_column(cols, ["outlet", "leave"], exclude=["condenser"])

        has_kw = kw_col is not None and wide_df[kw_col].notna().sum() > 30
        has_flow = flow_col is not None and wide_df[flow_col].notna().sum() > 30
        has_inlet = inlet_col is not None and wide_df[inlet_col].notna().sum() > 30
        has_outlet = outlet_col is not None and wide_df[outlet_col].notna().sum() > 30

        # Run validation agent
        flagged_df, report_df = validate(wide_df)

        def get_flag_pct(c_name):
            if not c_name or c_name not in report_df["column"].values:
                return np.nan
            return report_df.loc[report_df["column"] == c_name, "pct_flagged"].values[0]

        kw_flag = get_flag_pct(kw_col)
        flow_flag = get_flag_pct(flow_col)
        inlet_flag = get_flag_pct(inlet_col)
        outlet_flag = get_flag_pct(outlet_col)

        if not (has_kw and has_flow and has_inlet and has_outlet):
            status = "INCOMPLETE_UNIVERSAL_SET"
        elif (inlet_flag == 0.0 or np.isnan(inlet_flag)) and (outlet_flag == 0.0 or np.isnan(outlet_flag)) and (flow_flag < 5.0) and (kw_flag < 5.0):
            status = "CLEAN & VALID"
        else:
            status = "CORRUPTED / HIGH_FLAGS"

        records.append({
            "machineId": m_id,
            "n_rows": len(wide_df),
            "has_kw": has_kw,
            "has_flow": has_flow,
            "has_inlet": has_inlet,
            "has_outlet": has_outlet,
            "kw_flag_pct": kw_flag,
            "flow_flag_pct": flow_flag,
            "inlet_flag_pct": inlet_flag,
            "outlet_flag_pct": outlet_flag,
            "status": status
        })
        print(f"Chiller {m_id:4d} | Rows: {len(wide_df):5d} | Status: {status}", flush=True)

    res_df = pd.DataFrame(records)
    out_path = os.path.join(PROJECT_ROOT, "data", "clean_chillers_aug20.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved fleet scan to {out_path}", flush=True)

    # Summary
    print("\n" + "=" * 90, flush=True)
    print("AUG 2026 FLEET SCAN SUMMARY (FRESH FLEET)", flush=True)
    print("=" * 90, flush=True)
    print(res_df["status"].value_counts().to_string(), flush=True)

    clean_units = res_df[res_df["status"] == "CLEAN & VALID"]
    print(f"\nClean & Valid Chillers Count: {len(clean_units)} / {len(chillers)}", flush=True)
    print(clean_units[["machineId", "n_rows", "kw_flag_pct", "flow_flag_pct", "inlet_flag_pct", "outlet_flag_pct"]].to_string(index=False), flush=True)

    return res_df


if __name__ == "__main__":
    scan_chillers()
