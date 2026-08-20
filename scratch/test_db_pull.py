import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

app_conn = psycopg2.connect(
    host=os.environ["APPDB_HOST"],
    port=os.environ["APPDB_PORT"],
    dbname=os.environ["APPDB_NAME"],
    user=os.environ["APPDB_USER"],
    password=os.environ["APPDB_PASSWORD"]
)

query_sensors = """
    SELECT me."MachineId" as machine_id, me."Id" as sensor_id, me."SeriesDescription" as series_desc
    FROM "MachineExplorer" me
    JOIN machine m ON m."machineId" = me."MachineId"
    WHERE m."machineId" IN (4054, 2828, 2821) AND me."SeriesDescription" IS NOT NULL
"""
sensors_df = pd.read_sql_query(query_sensors, app_conn)
app_conn.close()

print("Sensors fetched:")
print(sensors_df.head(15))

ts_conn = psycopg2.connect(
    host=os.environ["TIMESCALE_HOST"],
    port=os.environ["TIMESCALE_PORT"],
    dbname=os.environ["TIMESCALE_NAME"],
    user=os.environ["TIMESCALE_USER"],
    password=os.environ["TIMESCALE_PASSWORD"]
)

sensor_ids = tuple(sensors_df["sensor_id"].tolist())
query_readings = """
    SELECT ("timestamp" AT TIME ZONE 'Asia/Calcutta') AS timestamp, machineexplorerid, value
    FROM trendseriesmeterdata
    WHERE machineexplorerid IN %s
    ORDER BY timestamp DESC
    LIMIT 1000
"""
readings_df = pd.read_sql_query(query_readings, ts_conn, params=(sensor_ids,))
ts_conn.close()

print(f"Readings fetched: {len(readings_df)}")
merged = readings_df.merge(sensors_df, left_on="machineexplorerid", right_on="sensor_id")
print("Merged head:")
print(merged.head())
