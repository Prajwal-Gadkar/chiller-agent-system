"""
Read-only exploration script: reports the earliest/latest timestamp and total
row count in trendseriesmeterdata across all chiller sensors, via a single
aggregate query (no row pulling). Useful for sizing a full pull before running
pull_trend_data.py over a large range.
"""

import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from list_chillers import QUERY as CHILLER_QUERY
from list_chillers import get_connection as get_appdb_connection

# NOTE: pd.read_sql_query over a raw (non-SQLAlchemy) psycopg2 connection
# silently converts tz-aware timestamptz values to UTC in the resulting
# DataFrame (same bug found and fixed in monthly_coverage.py). MIN/MAX are
# cast to local time via AT TIME ZONE in SQL so the values pandas receives
# are already plain local timestamps, never tz-aware in Python.
DB_TIMEZONE = "Asia/Calcutta"  # matches the Timescale session's `SHOW timezone`

RANGE_QUERY = """
    SELECT MIN("timestamp" AT TIME ZONE %(tz)s) AS earliest,
           MAX("timestamp" AT TIME ZONE %(tz)s) AS latest,
           COUNT(*) AS total_rows
    FROM trendseriesmeterdata
    WHERE machineexplorerid IN %(sensor_ids)s
"""


def get_timescale_connection():
    return psycopg2.connect(
        host=os.environ["TIMESCALE_HOST"],
        port=os.environ["TIMESCALE_PORT"],
        dbname=os.environ["TIMESCALE_NAME"],
        user=os.environ["TIMESCALE_USER"],
        password=os.environ["TIMESCALE_PASSWORD"],
    )


def load_chiller_sensor_ids():
    conn = get_appdb_connection()
    try:
        chiller_df = pd.read_sql_query(CHILLER_QUERY, conn)
    finally:
        conn.close()
    return chiller_df["MachineExplorerId"].dropna().unique().tolist()


def main():
    sensor_ids = load_chiller_sensor_ids()
    print(f"Chiller sensor Ids: {len(sensor_ids)}")

    conn = get_timescale_connection()
    try:
        result = pd.read_sql_query(RANGE_QUERY, conn, params={"sensor_ids": tuple(sensor_ids), "tz": DB_TIMEZONE})
    finally:
        conn.close()

    earliest = result.loc[0, "earliest"]
    latest = result.loc[0, "latest"]
    total_rows = int(result.loc[0, "total_rows"])

    span_days = (latest - earliest).days if pd.notna(earliest) and pd.notna(latest) else None

    print(f"Earliest timestamp: {earliest}")
    print(f"Latest timestamp:   {latest}")
    print(f"Total span:         {span_days} days")
    print(f"Total row count across all chiller sensors: {total_rows}")


if __name__ == "__main__":
    main()
