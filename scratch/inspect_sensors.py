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

for cid in [4054, 2828, 2821]:
    query_sensors = """
        SELECT me."MachineId" as machine_id, me."Id" as sensor_id, me."SeriesDescription" as series_desc
        FROM "MachineExplorer" me
        JOIN machine m ON m."machineId" = me."MachineId"
        WHERE m."machineId" = %s AND me."SeriesDescription" IS NOT NULL
    """
    df = pd.read_sql_query(query_sensors, app_conn, params=(cid,))
    print(f"--- Chiller {cid} ---")
    print(df["series_desc"].tolist())

app_conn.close()
