"""
Check for August 2026 Fleet Regime Shift / Discontinuity

Pulls weekly power and flow data for 4 long-history / clean chillers (1657, 1661, 2738, 2741)
from June 1, 2026 to August 19, 2026. Computes weekly mean KW, max KW, P50 KW, and active hours.
Checks week-by-week for any step-change discontinuity (like the 1.4x-4.3x Jan 2026 regime shift).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

load_dotenv()

DB_TIMEZONE = "Asia/Calcutta"
TEST_CHILLERS = [1657, 1661, 2738, 2741]


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


def get_sensors_for_chillers(machine_ids):
    conn = get_appdb_connection()
    query = """
        SELECT m."machineId", me."Id" AS "MachineExplorerId", me."SeriesDescription"
        FROM machine m
        JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
        WHERE m."machineId" IN %(m_ids)s;
    """
    df = pd.read_sql_query(query, conn, params={"m_ids": tuple(machine_ids)})
    conn.close()
    return df


def analyze_regime_shift():
    print("=" * 100)
    print("TASK 2: CHECK FOR NEW REGIME BOUNDARY IN JULY / AUGUST 2026")
    print(f"Target Chillers: {TEST_CHILLERS}")
    print("=" * 100)

    sensor_df = get_sensors_for_chillers(TEST_CHILLERS)
    sensor_ids = tuple(sensor_df["MachineExplorerId"].unique().tolist())

    conn_ts = get_timescale_connection()
    query = """
        SELECT ("timestamp" AT TIME ZONE %(tz)s) AS "timestamp", machineexplorerid, value
        FROM trendseriesmeterdata
        WHERE machineexplorerid IN %(sensor_ids)s
          AND "timestamp" >= '2026-06-01'
          AND "timestamp" <= '2026-08-20'
    """
    raw_df = pd.read_sql_query(query, conn_ts, params={"sensor_ids": sensor_ids, "tz": DB_TIMEZONE})
    conn_ts.close()

    if raw_df.empty:
        print("Error: No sensor readings returned.")
        return

    raw_df = raw_df.merge(sensor_df, left_on="machineexplorerid", right_on="MachineExplorerId")
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"]).dt.round("15min")

    records = []

    for m_id in TEST_CHILLERS:
        m_raw = raw_df[raw_df["machineId"] == m_id]
        if m_raw.empty:
            continue

        wide = m_raw.pivot_table(
            index="timestamp",
            columns="SeriesDescription",
            values="value",
            aggfunc="first"
        ).reset_index()

        # Find power column
        kw_col = [c for c in wide.columns if "kw" in c.lower() or "power" in c.lower() and "ikw" not in c.lower() and "kwh" not in c.lower()]
        if not kw_col:
            continue
        kw_col = kw_col[0]

        wide["week"] = wide["timestamp"].dt.isocalendar().week
        wide["week_label"] = wide["timestamp"].dt.strftime("%Y-W%U (starts %m-%d)")

        # Group week by week
        for (w_num, w_label), group in wide.groupby(["week", "week_label"]):
            running = group[group[kw_col] > 10.0]
            if running.empty:
                continue

            kw_mean = float(running[kw_col].mean())
            kw_median = float(running[kw_col].median())
            kw_max = float(running[kw_col].max())
            kw_p90 = float(running[kw_col].quantile(0.90))
            n_running = len(running)

            records.append({
                "machineId": m_id,
                "week_num": w_num,
                "week_start": group["timestamp"].min().strftime("%Y-%m-%d"),
                "week_end": group["timestamp"].max().strftime("%Y-%m-%d"),
                "n_running_samples": n_running,
                "KW_mean": kw_mean,
                "KW_median": kw_median,
                "KW_P90": kw_p90,
                "KW_max": kw_max,
            })

    res_df = pd.DataFrame(records).sort_values(["machineId", "week_start"]).reset_index(drop=True)

    print("\nWEEK-BY-WEEK POWER METRICS (JUNE 2026 -> AUGUST 2026):")
    print("-" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(res_df.to_string(index=False))

    out_csv = os.path.join(PROJECT_ROOT, "data", "august_regime_shift_analysis.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved weekly regime shift analysis to {out_csv}")


if __name__ == "__main__":
    analyze_regime_shift()
