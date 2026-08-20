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

ts_conn = psycopg2.connect(
    host=os.environ["TIMESCALE_HOST"],
    port=os.environ["TIMESCALE_PORT"],
    dbname=os.environ["TIMESCALE_NAME"],
    user=os.environ["TIMESCALE_USER"],
    password=os.environ["TIMESCALE_PASSWORD"]
)

chiller_ids = [4054, 2828, 2821]

for c_id in chiller_ids:
    query_sensors = """
        SELECT me."MachineId" as machine_id, me."Id" as sensor_id, me."SeriesDescription" as series_desc
        FROM "MachineExplorer" me
        JOIN machine m ON m."machineId" = me."MachineId"
        WHERE m."machineId" = %s AND me."SeriesDescription" IS NOT NULL
    """
    sensors_df = pd.read_sql_query(query_sensors, app_conn, params=(c_id,))
    sensor_ids = tuple(sensors_df["sensor_id"].tolist())
    
    query_ts = """
        SELECT DISTINCT ("timestamp" AT TIME ZONE 'Asia/Calcutta') AS timestamp
        FROM trendseriesmeterdata
        WHERE machineexplorerid IN %s
        ORDER BY timestamp DESC
        LIMIT 20
    """
    ts_df = pd.read_sql_query(query_ts, ts_conn, params=(sensor_ids,))
    print(f"Chiller {c_id}: distinct timestamps fetched = {len(ts_df)}")

app_conn.close()
ts_conn.close()
