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
      AND "timestamp" >= '2026-08-07' AND "timestamp" < '2026-08-08'
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

print("Comparison of Temperature Columns on 2026-08-07 for Chiller 2825:\n")
temp_cols = [c for c in wide_df.columns if "temp" in c.lower() or "inlet" in c.lower() or "outlet" in c.lower()]
print(wide_df[["timestamp"] + temp_cols].head(10).to_string(index=False))
