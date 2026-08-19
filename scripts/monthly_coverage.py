"""
Read-only exploration script: reports monthly row counts and distinct-sensor
coverage across all chiller sensors in trendseriesmeterdata, via a single
aggregate query. Useful for seeing at a glance whether data density is stable,
growing, or concentrated in a specific window before pulling a full range.
"""

import os
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from list_chillers import QUERY as CHILLER_QUERY
from list_chillers import get_connection as get_appdb_connection

MONTHLY_QUERY = """
    SELECT to_char(date_trunc('month', "timestamp"), 'YYYY-MM') AS month,
           COUNT(*) AS row_count,
           COUNT(DISTINCT machineexplorerid) AS distinct_sensors
    FROM trendseriesmeterdata
    WHERE machineexplorerid IN %(sensor_ids)s
    GROUP BY 1
    ORDER BY 1
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

    conn = get_timescale_connection()
    try:
        result = pd.read_sql_query(MONTHLY_QUERY, conn, params={"sensor_ids": tuple(sensor_ids)})
    finally:
        conn.close()

    pd.set_option("display.max_rows", None)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
