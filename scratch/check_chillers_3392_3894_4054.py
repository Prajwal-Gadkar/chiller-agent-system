import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn_app = psycopg2.connect(
    host=os.environ["APPDB_HOST"],
    port=os.environ["APPDB_PORT"],
    dbname=os.environ["APPDB_NAME"],
    user=os.environ["APPDB_USER"],
    password=os.environ["APPDB_PASSWORD"],
)

query_meta = """
    SELECT m."machineId", me."Id" AS "MachineExplorerId", me."SeriesDescription"
    FROM machine m
    JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
    WHERE m."machineId" IN (3392, 3894, 4054)
    ORDER BY me."SeriesDescription", m."machineId";
"""
df = pd.read_sql_query(query_meta, conn_app)
print("=== SENSOR METADATA FOR 3392, 3894, 4054 ===")
print(df.to_string())

conn_app.close()
