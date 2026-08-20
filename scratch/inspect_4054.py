import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.environ["APPDB_HOST"],
    port=os.environ["APPDB_PORT"],
    dbname=os.environ["APPDB_NAME"],
    user=os.environ["APPDB_USER"],
    password=os.environ["APPDB_PASSWORD"],
)

query = """
    SELECT m."machineId", m."machineType", m."status", me."Id" AS "MachineExplorerId", me."SeriesDescription"
    FROM machine m
    JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
    WHERE m."machineId" = 4054 OR me."SeriesDescription" ILIKE '%wet%bulb%'
    ORDER BY m."machineId", me."Id";
"""

df = pd.read_sql_query(query, conn)
conn.close()

print(df.to_string())
