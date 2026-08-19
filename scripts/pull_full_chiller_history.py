"""
Pulls full multi-year continuous time-series sensor history for priority chillers
(machineId 1657, 1660, 1661) from PostgreSQL databases and pivots to wide format.

- 1657: 2023-04-29 to 2026-07-08 (~1,166 days, ~1.97M rows raw)
- 1660: 2025-01-10 to 2026-07-08 (~544 days, ~900K rows raw)
- 1661: 2025-01-10 to 2026-07-08 (~544 days, ~971K rows raw)

Saves wide-format CSVs to data/chiller_<machineId>_full_history.csv.
Read-only on source DBs. No hardcoded secrets.
"""

import os
import sys
import argparse
import datetime
import dotenv
import psycopg2
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

dotenv.load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Database Credentials
APPDB_HOST = os.getenv("APPDB_HOST", "localhost")
APPDB_PORT = os.getenv("APPDB_PORT", "5432")
APPDB_NAME = os.getenv("APPDB_NAME", "Persistent_AppDb")
APPDB_USER = os.getenv("APPDB_USER", "postgres")
APPDB_PASSWORD = os.getenv("APPDB_PASSWORD", "admin")

TIMESCALE_HOST = os.getenv("TIMESCALE_HOST", "localhost")
TIMESCALE_PORT = os.getenv("TIMESCALE_PORT", "5432")
TIMESCALE_NAME = os.getenv("TIMESCALE_NAME", "Persistent_Timescale")
TIMESCALE_USER = os.getenv("TIMESCALE_USER", "postgres")
TIMESCALE_PASSWORD = os.getenv("TIMESCALE_PASSWORD", "admin")

TARGET_MACHINES = [1657, 1660, 1661]
CHUNK_DAYS = 30  # Pull in 30-day chunks per machine to bound memory usage

CHILLER_SENSOR_QUERY = """
SELECT 
    m."machineId",
    m."status",
    m."Criticality",
    me."Id" AS "MachineExplorerId",
    me."SeriesDescription"
FROM machine m
JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
WHERE m."machineId" = %s AND m."machineType" = 'Chiller';
"""

TREND_DATA_QUERY = """
SELECT 
    ("timestamp" AT TIME ZONE 'Asia/Calcutta')::text AS "timestamp",
    machineexplorerid,
    value
FROM trendseriesmeterdata
WHERE machineexplorerid IN %s
  AND "timestamp" >= %s::timestamptz
  AND "timestamp" < %s::timestamptz;
"""


def get_appdb_connection():
    return psycopg2.connect(
        host=APPDB_HOST, port=APPDB_PORT, dbname=APPDB_NAME,
        user=APPDB_USER, password=APPDB_PASSWORD
    )


def get_timescale_connection():
    return psycopg2.connect(
        host=TIMESCALE_HOST, port=TIMESCALE_PORT, dbname=TIMESCALE_NAME,
        user=TIMESCALE_USER, password=TIMESCALE_PASSWORD
    )


def pull_chiller_metadata(m_id):
    conn = get_appdb_connection()
    try:
        cur = conn.cursor()
        cur.execute(CHILLER_SENSOR_QUERY, (m_id,))
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=["machineId", "status", "Criticality", "MachineExplorerId", "SeriesDescription"])
        return df
    finally:
        conn.close()


def pivot_chunk(raw_df, meta_df, sensor_cols, machine_meta):
    """
    Round timestamp to nearest 15min -> pivot -> reindex to fixed columns -> merge metadata.
    """
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601").dt.round("15min")

    df = df.merge(
        meta_df[["MachineExplorerId", "machineId", "SeriesDescription"]],
        left_on="machineexplorerid",
        right_on="MachineExplorerId",
        how="left"
    )

    wide = df.pivot_table(
        index=["machineId", "timestamp"],
        columns="SeriesDescription",
        values="value",
        aggfunc="first"
    )

    wide = wide.reindex(columns=sensor_cols).reset_index()
    wide.columns.name = None

    wide = wide.merge(machine_meta, on="machineId", how="left")

    wide["machineId"] = wide["machineId"].astype("int32")
    for col in sensor_cols:
        wide[col] = wide[col].astype(np.float32)

    return wide[["machineId", "timestamp"] + sensor_cols + ["status", "Criticality"]]


def process_machine_history(m_id, out_csv_path):
    print(f"\n" + "="*80)
    print(f"PROCESSING FULL HISTORY FOR MACHINE {m_id}")
    print("="*80)

    meta_df = pull_chiller_metadata(m_id)
    if meta_df.empty:
        print(f"No metadata found for machineId {m_id}")
        return

    sensor_ids = tuple(int(x) for x in meta_df["MachineExplorerId"].unique())
    sensor_cols = sorted([str(c) for c in meta_df["SeriesDescription"].dropna().unique().tolist()])
    machine_meta = meta_df[["machineId", "status", "Criticality"]].drop_duplicates().iloc[0:1]

    print(f"Machine {m_id}: {len(sensor_ids)} sensors defined in AppDb.")
    print(f"Sensors: {sensor_cols}")

    # Determine date range for this machine in TimescaleDB
    ts_conn = get_timescale_connection()
    try:
        cur = ts_conn.cursor()
        cur.execute("""
            SELECT MIN("timestamp" AT TIME ZONE 'Asia/Calcutta'), MAX("timestamp" AT TIME ZONE 'Asia/Calcutta')
            FROM trendseriesmeterdata
            WHERE machineexplorerid IN %s;
        """, (sensor_ids,))
        min_ts, max_ts = cur.fetchone()
    finally:
        ts_conn.close()

    if not min_ts or not max_ts:
        print(f"No time-series data found in TimescaleDB for machineId {m_id}")
        return

    print(f"Available DB Date Range: {min_ts} to {max_ts}")

    start_date = min_ts.date()
    end_date = max_ts.date() + datetime.timedelta(days=1)

    # Chunked extraction and streaming to CSV
    cur_start = start_date
    first_chunk = True
    total_written_rows = 0

    if os.path.exists(out_csv_path):
        os.remove(out_csv_path)

    ts_conn = get_timescale_connection()
    try:
        while cur_start < end_date:
            cur_end = min(cur_start + datetime.timedelta(days=CHUNK_DAYS), end_date)
            start_str = f"{cur_start.strftime('%Y-%m-%d')} 00:00:00+05:30"
            end_str = f"{cur_end.strftime('%Y-%m-%d')} 00:00:00+05:30"

            cur = ts_conn.cursor()
            cur.execute(TREND_DATA_QUERY, (sensor_ids, start_str, end_str))
            rows = cur.fetchall()
            cur.close()

            if rows:
                raw_chunk_df = pd.DataFrame(rows, columns=["timestamp", "machineexplorerid", "value"])
                wide_chunk_df = pivot_chunk(raw_chunk_df, meta_df, sensor_cols, machine_meta)

                if not wide_chunk_df.empty:
                    wide_chunk_df.to_csv(out_csv_path, mode="a", header=first_chunk, index=False)
                    first_chunk = False
                    total_written_rows += len(wide_chunk_df)

            print(f"  Chunk {cur_start} -> {cur_end}: {len(rows):,} raw rows pulled.")
            cur_start = cur_end
    finally:
        ts_conn.close()

    # Load summary of written CSV
    if os.path.exists(out_csv_path):
        out_df = pd.read_csv(out_csv_path)
        min_p = out_df["timestamp"].min()
        max_p = out_df["timestamp"].max()
        print(f"\nSUCCESS: Saved {out_csv_path}")
        print(f"  Pivoted Wide Rows: {len(out_df):,}")
        print(f"  Columns ({len(out_df.columns)}): {out_df.columns.tolist()[:6]} ...")
        print(f"  Actual Time-Series Range: {min_p} to {max_p}")


def main():
    parser = argparse.ArgumentParser(description="Pull full time-series history for chillers 1657, 1660, 1661.")
    args = parser.parse_args()

    for m_id in TARGET_MACHINES:
        out_csv = os.path.join(PROJECT_ROOT, "data", f"chiller_{m_id}_full_history.csv")
        process_machine_history(m_id, out_csv)


if __name__ == "__main__":
    main()
